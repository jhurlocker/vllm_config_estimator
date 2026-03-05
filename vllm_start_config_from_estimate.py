#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import math
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


# ============================================================
# Utilities & Hugging Face Integration
# ============================================================


def which(cmd: str) -> Optional[str]:
    """
    Locates the executable path for a given command in the system PATH.

    Args:
        cmd: The name of the command to locate.

    Returns:
        The absolute path to the executable if found, otherwise None.
    """
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(d) / cmd
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def run_cmd(
    cmd: List[str], timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    """
    Executes a shell command safely using subprocess.run.

    Args:
        cmd: A list of command arguments.
        timeout: Optional timeout in seconds for the command execution.

    Returns:
        A subprocess.CompletedProcess instance containing the execution result.
    """
    return subprocess.run(
        cmd, text=True, capture_output=True, timeout=timeout, check=False
    )


def shell_join(args: List[str]) -> str:
    """
    Joins a list of command arguments into a single shell-escaped string.

    Args:
        args: A list of command arguments.

    Returns:
        A shell-escaped string suitable for execution or display.
    """
    return " ".join(shlex.quote(a) for a in args)


def format_args_multiline(args: List[str], joiner: str) -> str:
    """
    Formats a list of command-line arguments into a multiline string for better readability.
    Groups options (e.g., --flag) with their corresponding values.

    Args:
        args: A list of command-line arguments.
        joiner: The string used to join the formatted lines.

    Returns:
        A formatted, multiline string representation of the arguments.
    """
    lines = []
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--"):
            line_parts = [token]
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                line_parts.append(shlex.quote(args[i]))
                i += 1
            lines.append(" ".join(line_parts))
        else:
            lines.append(shlex.quote(token))
            i += 1
    return joiner.join(lines)


def fetch_hf_config(model: str) -> Optional[Dict]:
    """
    Fetches the config.json for a specified model from the Hugging Face Hub.

    Args:
        model: The Hugging Face model identifier (e.g., 'meta-llama/Llama-2-7b-hf').

    Returns:
        A dictionary containing the parsed JSON configuration if successful, otherwise None.
    """
    url = f"https://huggingface.co/{model}/resolve/main/config.json"
    headers = {"User-Agent": "llm-d-pipeline-validator"}

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(
            f"WARN: HF Config fetch failed ({e.code} {e.reason}) - fallback heuristics will be used.",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"WARN: HF Config fetch failed ({e}) - fallback heuristics will be used.",
            file=sys.stderr,
        )

    return None


def is_moe_model(model: str, config: Optional[Dict]) -> bool:
    """
    Determines if a model is a Mixture of Experts (MoE) architecture based on its name or Hugging Face config.

    Args:
        model: The model name or identifier.
        config: The optional Hugging Face configuration dictionary.

    Returns:
        True if the model is identified as an MoE, False otherwise.
    """
    if config:
        moe_keys = [
            "num_experts_per_tok",
            "num_local_experts",
            "experts_per_tok",
            "moe_num_experts",
            "n_routed_experts",
        ]
        if any(key in config for key in moe_keys):
            return True
        archs = config.get("architectures", [])
        model_type = config.get("model_type", "").lower()
        if any("moe" in arch.lower() or "mixtral" in arch.lower() for arch in archs):
            return True
        if "moe" in model_type or "mixtral" in model_type:
            return True
        return False

    m = model.lower()
    if any(k in m for k in ["moe", "mixtral", "deepseek"]):
        return True
    if re.search(r"-a\d+b", m):
        return True
    return False


def get_hf_max_context(config: Optional[Dict]) -> Optional[int]:
    """
    Extracts the maximum context window size from a Hugging Face configuration dictionary.

    Args:
        config: The Hugging Face configuration dictionary.

    Returns:
        The maximum sequence length as an integer if found, otherwise None.
    """
    if not config:
        return None
    for key in ["max_position_embeddings", "max_sequence_length", "seq_length"]:
        if key in config and isinstance(config[key], int):
            return config[key]
    return None


def detect_quantization_from_config(config: Optional[Dict]) -> Optional[str]:
    """
    Attempts to detect the quantization scheme (e.g. 'int4', 'fp8', 'awq') directly
    from the Hugging Face config.json if present.

    Args:
        config: The Hugging Face configuration dictionary.

    Returns:
        A string representing the detected quantization format, or None if not found/recognized.
    """
    if not config:
        return None

    q_config = config.get("quantization_config", {})
    if not q_config:
        return None

    method = q_config.get("quant_method", "").lower()
    bits = q_config.get("bits", None)

    if method == "fp8":
        return "fp8"
    if method in ["awq", "gptq"]:
        return "int4" if bits == 4 else method

    if method == "compressed-tensors":
        groups = q_config.get("config_groups", {})
        for _, group_info in groups.items():
            weights = group_info.get("weights", {})
            num_bits = weights.get("num_bits")
            q_type = weights.get("type", "").lower()

            if q_type == "int" and num_bits == 4:
                return "int4"
            if q_type == "int" and num_bits == 8:
                return "int8"
            if q_type == "float" and num_bits == 8:
                return "fp8"

    return None


def normalize_gpu_name(gpu: str) -> str:
    """
    Normalizes a GPU string identifier to a standard canonical name.

    Args:
        gpu: The raw GPU name string (e.g., 'a100-80g', 'RTX 4090').

    Returns:
        The normalized uppercase string representation.
    """
    s = gpu.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "b200": "B200",
        "b100": "B100",
        "h200": "H200",
        "h100": "H100",
        "h100sxm": "H100",
        "h100pcie": "H100",
        "a100": "A100",
        "a10080g": "A100",
        "a10040g": "A100",
        "a6000": "A6000",
        "a6000ada": "A6000",
        "a40": "A40",
        "l40s": "L40S",
        "l40": "L40",
        "a10g": "A10G",
        "a10": "A10G",
        "l4": "L4",
        "t4": "T4",
        "v100": "V100",
        "rtx4090": "RTX4090",
        "rtx3090": "RTX3090",
        "mi300x": "MI300X",
        "mi250": "MI250",
    }
    return aliases.get(s, gpu)


def infer_gpu_memory_gb(gpu: str) -> Optional[int]:
    """
    Infers the estimated memory capacity in Gigabytes for a given GPU name.

    Args:
        gpu: The raw GPU name string.

    Returns:
        The integer memory size in GB if recognized, otherwise None.
    """
    s = gpu.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "b200": 192,
        "b100": 192,
        "h200": 141,
        "h100": 80,
        "a100": 80,
        "a10080g": 80,
        "a10040g": 40,
        "a6000": 48,
        "a6000ada": 48,
        "a40": 48,
        "l40s": 48,
        "l40": 48,
        "a10g": 24,
        "a10": 24,
        "l4": 24,
        "t4": 16,
        "v100": 32,
        "v10016g": 16,
        "v10032g": 32,
        "rtx4090": 24,
        "rtx3090": 24,
        "mi300x": 192,
        "mi250": 128,
    }
    return mapping.get(s)


def parse_param_count_billions(
    model: str, config: Optional[Dict] = None
) -> Optional[float]:
    """
    Estimates the number of parameters in billions for a given model.
    Attempts to parse from the model name first, then infers from the Hugging Face config.

    Args:
        model: The model name or identifier.
        config: The optional Hugging Face configuration dictionary.

    Returns:
        The estimated parameter count in billions as a float, or None if it cannot be determined.
    """
    # 1. Try regex on model string
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])\b", model)
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2).upper()
            if unit == "M":
                return value / 1000.0
            return value
        except Exception:
            pass

    # 2. Try inferring from HF Config architecture hidden size / layers if available
    # A very rough approximation for missing parameter sizes
    if config:
        # If parameters count is explicitly stored somewhere (rare but possible)
        if "num_parameters" in config:
            return float(config["num_parameters"]) / 1e9

        hidden_size = config.get("hidden_size", 0)
        num_layers = config.get("num_hidden_layers", 0)
        vocab_size = config.get("vocab_size", 0)

        if hidden_size and num_layers:
            # Dense model approximation: (12 * h^2) * L + (V * h)
            # MoE model approximation is much harder without active experts count, but we can guess
            is_moe = is_moe_model(model, config)
            if is_moe:
                num_experts = config.get(
                    "num_local_experts", config.get("moe_num_experts", 8)
                )
                # rough MoE param estimate (base dense + experts)
                params = (
                    (12 * (hidden_size**2)) * num_layers * (num_experts * 0.5)
                ) + (vocab_size * hidden_size)
            else:
                params = (12 * (hidden_size**2) * num_layers) + (
                    vocab_size * hidden_size
                )

            if params > 0:
                return params / 1e9

    return None


# ============================================================
# llm-optimizer estimate wrapper
# ============================================================


@dataclass
class EstimateInputs:
    """
    Data class representing the input parameters required to run the llm-optimizer estimate command.
    """

    model: str
    gpu: str
    num_gpus: int
    num_nodes: int
    input_len: int
    output_len: int
    constraints: Optional[str] = None
    target: Optional[str] = None
    extra_args: Optional[List[str]] = None


@dataclass
class EstimateResult:
    """
    Data class representing the output and parsed metrics from an llm-optimizer estimate execution.
    """

    returncode: int
    stdout: str
    stderr: str
    parsed: Dict[str, float]


def build_estimate_cmd(inp: EstimateInputs) -> List[str]:
    """
    Constructs the command-line arguments for the llm-optimizer estimate tool.

    Args:
        inp: An EstimateInputs instance containing the configuration.

    Returns:
        A list of strings representing the command and its arguments.
    """
    cmd = [
        "llm-optimizer",
        "estimate",
        "--model",
        inp.model,
        "--gpu",
        normalize_gpu_name(inp.gpu),
        "--num-gpus",
        str(inp.num_gpus),
        "--input-len",
        str(inp.input_len),
        "--output-len",
        str(inp.output_len),
    ]
    if inp.constraints:
        cmd += ["--constraints", inp.constraints]
    if inp.target:
        cmd += ["--target", inp.target]
    if inp.extra_args:
        cmd += inp.extra_args
    return cmd


def parse_estimate_output(text: str) -> Dict[str, float]:
    """
    Parses the standard output from the llm-optimizer estimate command to extract key performance metrics.

    Args:
        text: The raw output text from the command.

    Returns:
        A dictionary mapping metric names (e.g., 'ttft_ms', 'output_tps') to their float values.
    """
    metrics: Dict[str, float] = {}
    patterns = [
        (r"TTFT:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", "ttft_ms"),
        (r"ITL:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", "itl_ms"),
        (r"E2E:\s*([0-9]+(?:\.[0-9]+)?)\s*s", "e2e_s"),
        (r"Output:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", "output_tps"),
        (r"Input:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", "input_tps"),
        (r"Requests:\s*([0-9]+(?:\.[0-9]+)?)\s*req/s", "req_s"),
        (r"Concurrency:\s*([0-9]+(?:\.[0-9]+)?)", "concurrency"),
        (r"concurrency=([0-9]+(?:\.[0-9]+)?)", "concurrency"),
    ]
    for pat, key in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except Exception:
                pass
    return metrics


def run_estimate(inp: EstimateInputs, timeout: int = 300) -> EstimateResult:
    """
    Executes the llm-optimizer estimate command and parses its output.

    Args:
        inp: The input parameters for the estimation.
        timeout: Execution timeout in seconds.

    Returns:
        An EstimateResult instance containing the command output and parsed metrics.
    """
    cmd = build_estimate_cmd(inp)
    p = run_cmd(cmd, timeout=timeout)
    combined = (p.stdout or "") + "\n" + (p.stderr or "")
    parsed = parse_estimate_output(combined)
    return EstimateResult(
        returncode=p.returncode,
        stdout=p.stdout or "",
        stderr=p.stderr or "",
        parsed=parsed,
    )


# ============================================================
# Model-specific presets + candidates
# ============================================================


@dataclass
class CandidateConfig:
    """
    Data class representing a proposed vLLM configuration profile.
    Contains the arguments, reasoning, tuning suggestions, and a corresponding guidellm command.
    """

    name: str
    args: List[str]
    rationale: Dict[str, str]
    tuning_knobs: List[str]
    guidellm_cmd: str


def detect_model_family(
    model: str, override: Optional[str], config: Optional[Dict]
) -> str:
    """
    Determines the broad model family to apply family-specific default configurations.

    Args:
        model: The model name or identifier.
        override: An optional explicit family string provided by the user.
        config: The Hugging Face configuration dictionary.

    Returns:
        A string representing the identified model family (e.g., 'gpt-oss', 'llama', 'qwen').
    """
    if override:
        return override.lower()

    m_type = config.get("model_type", "").lower() if config else ""
    m_name = model.lower()

    if "gpt-oss" in m_name or "gpt-oss" in m_type:
        return "gpt-oss"
    if "llama" in m_type or "llama" in m_name:
        return "llama"
    if "qwen" in m_type or "qwen" in m_name:
        return "qwen"
    if "granite" in m_type or "granite" in m_name:
        return "granite"
    return "default"


def model_family_defaults(family: str) -> Dict[str, object]:
    """
    Provides default vLLM configuration biases based on the model family.

    Args:
        family: The identified model family string.

    Returns:
        A dictionary containing default settings for expert parallelism, remote code execution, and prefix caching.
    """
    base = {
        "enable_expert_parallel_default": None,
        "trust_remote_code_default": True,
        "prefix_caching_bias": False,
    }
    if family == "gpt-oss":
        base.update(
            {
                "enable_expert_parallel_default": True,
                "trust_remote_code_default": True,
                "prefix_caching_bias": True,
            }
        )
    elif family == "llama":
        base.update(
            {
                "enable_expert_parallel_default": False,
                "trust_remote_code_default": False,
                "prefix_caching_bias": True,
            }
        )
    elif family == "qwen":
        base.update(
            {
                "enable_expert_parallel_default": False,
                "trust_remote_code_default": True,
                "prefix_caching_bias": True,
            }
        )
    elif family == "granite":
        base.update(
            {
                "enable_expert_parallel_default": False,
                "trust_remote_code_default": True,
                "prefix_caching_bias": False,
            }
        )
    return base


def infer_tp_pp_dp(
    num_gpus: int,
    num_nodes: int,
    model: str,
    gpu: str,
    dtype: str,
    quantization: Optional[str],
    candidate: str,
    hf_config: Optional[Dict],
    model_params_b_override: Optional[float] = None,
) -> Tuple[int, int, int]:
    """
    Calculates the optimal 3D Parallelism configuration (Tensor, Pipeline, and Data Parallelism)
    constrained by cluster topology and memory requirements.

    Args:
        num_gpus: Total number of GPUs available.
        num_nodes: Number of physical nodes.
        model: The model name or identifier.
        gpu: The GPU type/name.
        dtype: Data type (e.g., 'float16', 'bfloat16').
        quantization: Optional quantization scheme.
        candidate: Target profile ('latency', 'throughput', or 'balanced').
        hf_config: Optional Hugging Face configuration dictionary.
        model_params_b_override: Optional explicit parameter count override in billions.

    Returns:
        A tuple of (tensor_parallel_size, pipeline_parallel_size, data_parallel_size).
    """
    mem = infer_gpu_memory_gb(gpu)
    param_b = (
        model_params_b_override
        if model_params_b_override is not None
        else parse_param_count_billions(model, hf_config)
    )
    weights_gb = estimate_weight_memory_gb(param_b, dtype, quantization)

    gpus_per_node = max(1, num_gpus // max(1, num_nodes))

    if mem is None or weights_gb is None:
        # Fallback to pure TP if single node, or TP+PP if multi-node
        return gpus_per_node, num_nodes, 1

    # Find the minimum number of GPUs required to fit the model weights + overhead
    min_shards = math.ceil(weights_gb / (mem * 0.85))

    valid_topologies = []

    # Generate all valid TP/PP/DP combinations
    # Rule 1: TP must not cross node boundaries
    for tp in range(1, gpus_per_node + 1):
        if gpus_per_node % tp != 0:
            continue

        for pp in range(1, num_gpus + 1):
            if (tp * pp) < min_shards:
                continue
            if num_gpus % (tp * pp) != 0:
                continue

            dp = num_gpus // (tp * pp)
            valid_topologies.append((tp, pp, dp))

    # If the model simply won't fit on the provided hardware, return max capacity
    if not valid_topologies:
        return gpus_per_node, num_nodes, 1

    if candidate == "latency":
        # Prioritize compute speed: Maximize TP, then PP. Ignore DP.
        valid_topologies.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return valid_topologies[0]

    elif candidate == "throughput":
        # Prioritize replicas: Maximize DP. Use only as much TP/PP as strictly necessary.
        valid_topologies.sort(key=lambda x: (x[2], x[0]), reverse=True)
        return valid_topologies[0]

    else:
        # Balanced: Sort by DP, find the middle ground
        valid_topologies.sort(key=lambda x: (x[2], x[0]))
        return valid_topologies[len(valid_topologies) // 2]


def choose_gpu_memory_utilization(gpu: str, total_ctx: int, candidate: str) -> float:
    """
    Heuristically determines the ideal gpu_memory_utilization fraction for vLLM based on
    GPU VRAM, total context length, and the target performance profile.

    Args:
        gpu: The GPU type/name.
        total_ctx: The total expected context window (input + output length).
        candidate: Target profile ('latency', 'throughput', or 'balanced').

    Returns:
        A float value representing the suggested gpu_memory_utilization (e.g., 0.90).
    """
    mem = infer_gpu_memory_gb(gpu)
    if mem is None:
        base = 0.90
    elif mem >= 80:
        base = 0.92
    elif mem >= 48:
        base = 0.90
    else:
        base = 0.88

    if total_ctx > 16000:
        base -= 0.02
    elif total_ctx > 8000:
        base -= 0.01

    if candidate == "latency":
        base -= 0.02
    elif candidate == "throughput":
        base += 0.03

    return max(0.82, min(0.95, round(base, 2)))


def choose_max_num_seqs(
    gpu: str,
    num_gpus: int,
    tp: int,
    pp: int,
    total_ctx: int,
    candidate: str,
    est: Dict[str, float],
    model: str,
    dtype: str,
    quantization: Optional[str],
    hf_config: Optional[Dict],
    model_params_b_override: Optional[float],
) -> int:
    """
    Determines an appropriate value for max_num_seqs (batch size limit in requests) based on
    hardware capacity, model size, and expected context length.

    Args:
        gpu: The GPU type/name.
        num_gpus: Total number of GPUs.
        tp: Tensor parallel size.
        pp: Pipeline parallel size.
        total_ctx: Total context length.
        candidate: Target profile ('latency', 'throughput', or 'balanced').
        est: Parsed metrics from llm-optimizer estimate.
        model: The model name or identifier.
        dtype: The data type.
        quantization: Optional quantization scheme.
        hf_config: Hugging Face config.
        model_params_b_override: Explicit parameter override.

    Returns:
        The suggested maximum number of concurrent sequences.
    """
    mem = infer_gpu_memory_gb(gpu)

    if (mem or 0) >= 80 and num_gpus >= 4:
        base = 64
    elif (mem or 0) >= 48:
        base = 48
    else:
        base = 24

    if total_ctx > 16000:
        base = max(12, base // 2)
    elif total_ctx > 8000:
        base = max(16, int(base * 0.75))

    param_b = (
        model_params_b_override
        if model_params_b_override is not None
        else parse_param_count_billions(model, hf_config)
    )
    weights_gb_total = estimate_weight_memory_gb(param_b, dtype, quantization)

    if mem is not None and weights_gb_total is not None:
        weights_per_gpu = weights_gb_total / max(tp * pp, 1)
        usable_vram_per_gpu = mem * 0.90
        kv_cache_budget = usable_vram_per_gpu - weights_per_gpu

        if kv_cache_budget < 8.0:
            if kv_cache_budget <= 0:
                base = 4
            else:
                safe_cap = max(4, int(kv_cache_budget * 3))
                base = min(base, safe_cap)

    if candidate == "latency":
        base = max(4, int(base * 0.5))
    elif candidate == "throughput":
        base = int(base * 1.5)

    base = min(base, 128)

    conc = int(est.get("concurrency", 0) or 0)
    if conc > 0:
        if candidate == "throughput":
            base = min(max(base, min(conc // 2, 128)), 160)
        elif candidate == "latency":
            base = min(base, max(4, conc // 4))

    return int(base)


def choose_max_num_batched_tokens(
    gpu: str, num_gpus: int, total_ctx: int, candidate: str
) -> int:
    """
    Determines an optimal max_num_batched_tokens value (for chunked prefill)
    based on hardware capacity and the target optimization profile.

    Args:
        gpu: The GPU type/name.
        num_gpus: Total number of GPUs.
        total_ctx: Total context length.
        candidate: Target profile ('latency', 'throughput', or 'balanced').

    Returns:
        The suggested max_num_batched_tokens integer value.
    """
    mem = infer_gpu_memory_gb(gpu)
    if (mem or 0) >= 80:
        base = 4096
    elif (mem or 0) >= 48:
        base = 2048
    else:
        base = 1024

    if candidate == "latency":
        base = max(512, int(base * 0.5))
    elif candidate == "throughput":
        base = min(8192, int(base * 2))
    return int(base)


def choose_stream_interval(candidate: str, prefer_streaming_smoothness: bool) -> int:
    """
    Chooses the stream interval for token generation responses.

    Args:
        candidate: Target profile ('latency', 'throughput', or 'balanced').
        prefer_streaming_smoothness: Flag to force a stream interval of 1 for smooth text generation.

    Returns:
        The suggested stream_interval integer.
    """
    if prefer_streaming_smoothness:
        return 1
    return {"latency": 1, "balanced": 5, "throughput": 10}[candidate]


def choose_prefill_partials(candidate: str) -> Tuple[int, int]:
    """
    Selects parameters for partial prefill scheduling to balance prompt processing vs generation.

    Args:
        candidate: Target profile ('latency', 'throughput', or 'balanced').

    Returns:
        A tuple of (max_num_partial_prefills, max_long_partial_prefills).
    """
    return {
        "latency": (2, 1),
        "balanced": (4, 1),
        "throughput": (8, 2),
    }[candidate]


def choose_async_scheduling(candidate: str, strict_ttft: bool) -> bool:
    """
    Decides whether async scheduling should be enabled based on the target profile and latency constraints.

    Args:
        candidate: Target profile ('latency', 'throughput', or 'balanced').
        strict_ttft: Whether strict Time-To-First-Token constraints are present.

    Returns:
        A boolean indicating whether to enable async scheduling.
    """
    if strict_ttft and candidate == "latency":
        return False
    return candidate in ("balanced", "throughput")


def choose_max_model_len(
    user_value: Optional[int],
    input_len: int,
    output_len: int,
    hf_max_ctx: Optional[int],
) -> int:
    """
    Computes a practical max_model_len constraint based on requested input/output sizes and model limits,
    snapping to power-of-two boundaries for better cache efficiency where possible.

    Args:
        user_value: Explicit user-provided max_model_len override.
        input_len: Expected maximum input length.
        output_len: Expected maximum output length.
        hf_max_ctx: Model's native maximum context window from its config.

    Returns:
        The suggested max_model_len limit.
    """
    if user_value:
        return user_value

    target = max(4096, int((input_len + output_len) * 2))

    if hf_max_ctx:
        if target > hf_max_ctx:
            target = hf_max_ctx
        if target < (input_len + output_len):
            target = input_len + output_len

    for snap in (4096, 8192, 16384, 32768, 65536, 131072):
        if target <= snap:
            return snap

    return target


def parse_constraints_for_ttft(constraints: Optional[str]) -> bool:
    """
    Checks if Time-To-First-Token (TTFT) constraints are explicitly specified in the user constraints string.

    Args:
        constraints: The raw constraints string.

    Returns:
        True if a TTFT constraint is detected, False otherwise.
    """
    return bool(constraints and "ttft" in constraints.lower())


# ============================================================
# Validation / feasibility checks
# ============================================================


@dataclass
class ValidationIssue:
    """
    Data class representing a detected configuration or feasibility issue.
    """

    level: str
    code: str
    message: str


def normalize_quantization(q: Optional[str]) -> Optional[str]:
    """
    Normalizes a quantization string identifier to a standard canonical name.

    Args:
        q: The raw quantization string (e.g., '8bit', 'awq', 'float16').

    Returns:
        The normalized quantization string, or None if none was provided.
    """
    if not q:
        return None
    s = q.strip().lower()
    aliases = {
        "none": None,
        "fp16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp8": "fp8",
        "float8": "fp8",
        "fp4": "fp4",
        "float4": "fp4",
        "nvfp4": "fp4",
        "int8": "int8",
        "8bit": "int8",
        "int4": "int4",
        "4bit": "int4",
        "awq": "int4",
        "gptq": "int4",
    }
    return aliases.get(s, s)


def effective_bytes_per_param(dtype: str, quantization: Optional[str]) -> float:
    """
    Estimates the effective number of bytes required per model parameter
    based on the data type and quantization scheme.

    Args:
        dtype: The raw data type string.
        quantization: Optional quantization scheme.

    Returns:
        A float representing the effective bytes per parameter (e.g., 2.0 for fp16, 0.5 for int4).
    """
    q = normalize_quantization(quantization)
    if q == "int4":
        return 0.5
    if q == "fp4":
        return 0.5
    if q == "int8":
        return 1.0
    if q == "fp8":
        return 1.0
    if q in ("bf16", "fp16"):
        return 2.0

    d = (dtype or "auto").lower()
    if d in ("bf16", "bfloat16", "fp16", "float16", "half", "auto"):
        return 2.0
    if d in ("fp8", "float8"):
        return 1.0
    return 2.0


def estimate_weight_memory_gb(
    param_b: Optional[float], dtype: str, quantization: Optional[str]
) -> Optional[float]:
    """
    Estimates the total memory required to load the model weights in Gigabytes,
    including a small overhead factor.

    Args:
        param_b: The number of model parameters in billions.
        dtype: The data type.
        quantization: Optional quantization scheme.

    Returns:
        The estimated memory in GB, or None if parameters count is unknown.
    """
    if param_b is None:
        return None
    bpp = effective_bytes_per_param(dtype, quantization)
    overhead_factor = 1.15
    return param_b * bpp * overhead_factor


def fetch_valid_vllm_args(version: str) -> Optional[Set[str]]:
    """
    Fetches the valid CLI arguments for a specific vLLM version from GitHub.

    Args:
        version: The vLLM version string (e.g., '0.11.2', 'v0.10.0').

    Returns:
        A set of valid argument strings (e.g., {'--max-model-len', '--dtype'}),
        or None if they could not be fetched.
    """
    if not version.startswith("v") and not version == "main":
        version = f"v{version}"

    files_to_check = [
        "vllm/engine/arg_utils.py",
        "vllm/entrypoints/openai/cli_args.py",
    ]

    all_args = set()
    success = False

    for f in files_to_check:
        url = f"https://raw.githubusercontent.com/vllm-project/vllm/{version}/{f}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode("utf-8")

                # 1. Match explicit string arguments like parser.add_argument("--foo")
                matches = re.findall(r"[\'\"](--[a-zA-Z0-9\-]+)[\'\"]", content)
                for match in matches:
                    all_args.add(match)

                # 2. In older vLLM versions (like < 0.6.0), engine arguments were primarily defined
                # as @dataclass fields which were then programmatically converted to CLI args.
                # E.g. max_num_seqs: int = 256 -> --max-num-seqs
                fields = re.findall(
                    r"^\s+([a-zA-Z0-9_]+)\s*:\s*[a-zA-Z0-9_\[\]\s,]+(?:=.*)?$",
                    content,
                    re.MULTILINE,
                )
                for field in fields:
                    flag = f"--{field.replace('_', '-')}"
                    all_args.add(flag)

                success = True
        except Exception:
            pass

    return all_args if success else None


def validate_feasibility(
    *,
    model: str,
    family: str,
    gpu: str,
    num_gpus: int,
    num_nodes: int,
    input_len: int,
    output_len: int,
    dtype: str,
    quantization: Optional[str],
    tp: int,
    pp: int,
    max_model_len: int,
    hf_config: Optional[Dict],
    hf_max_ctx: Optional[int],
    model_params_b_override: Optional[float] = None,
    candidate_args: Optional[List[str]] = None,
    valid_vllm_args: Optional[Set[str]] = None,
) -> List[ValidationIssue]:
    """
    Performs comprehensive feasibility checks on the proposed configuration,
    verifying hardware limits, context capacities, and distributed topology constraints.

    Args:
        model: Model identifier.
        family: Model family.
        gpu: GPU type.
        num_gpus: Total number of GPUs.
        num_nodes: Total number of nodes.
        input_len: Expected input token length.
        output_len: Expected output token length.
        dtype: Data type.
        quantization: Optional quantization string.
        tp: Tensor parallel size.
        pp: Pipeline parallel size.
        max_model_len: Configured max model length.
        hf_config: Hugging Face configuration dictionary.
        hf_max_ctx: Extracted max context from Hugging Face config.
        model_params_b_override: Override for model parameter count.

    Returns:
        A list of ValidationIssue objects containing errors, warnings, or info messages.
    """
    issues: List[ValidationIssue] = []

    if num_gpus < 1:
        issues.append(
            ValidationIssue("error", "NUM_GPUS_INVALID", "num_gpus must be >= 1")
        )
    if candidate_args and valid_vllm_args:
        for arg in candidate_args:
            if arg.startswith("--"):
                base_arg = arg.split("=")[0]
                if base_arg not in valid_vllm_args:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "INVALID_ENGINE_ARG",
                            f"Argument '{base_arg}' is not recognized in the target vLLM version.",
                        )
                    )

    if tp < 1:
        issues.append(
            ValidationIssue("error", "TP_INVALID", "tensor_parallel_size must be >= 1")
        )
    if pp < 1:
        issues.append(
            ValidationIssue(
                "error", "PP_INVALID", "pipeline_parallel_size must be >= 1"
            )
        )

    if (tp * pp) > num_gpus:
        issues.append(
            ValidationIssue(
                "error",
                "PARALLELISM_EXCEEDS_GPUS",
                f"TP*PP ({tp*pp}) exceeds total num_gpus ({num_gpus})",
            )
        )

    if pp > 1 and num_nodes > 1:
        issues.append(
            ValidationIssue(
                "info",
                "MULTI_NODE_PIPELINE_PARALLEL",
                f"Pipeline Parallelism (PP={pp}) across {num_nodes} nodes detected. This requires high-speed interconnects (like InfiniBand or RoCE) between nodes to avoid severe latency bottlenecks.",
            )
        )

    if tp > 1 and num_nodes > 1 and (tp * pp > num_gpus):
        # Skip evaluating this specific check here, handled below
        pass
    elif tp > 1 and num_nodes > 1 and tp > (num_gpus // num_nodes):
        # Only emit this warning if TP actually crosses a node boundary
        issues.append(
            ValidationIssue(
                "warning",
                "MULTI_NODE_TENSOR_PARALLEL",
                f"Tensor Parallelism (TP={tp}) across nodes detected. TP across nodes requires extremely high-bandwidth, ultra-low latency networking (e.g., NVLink over InfiniBand) to perform acceptably.",
            )
        )

    if not hf_config and model_params_b_override is None:
        issues.append(
            ValidationIssue(
                "warning",
                "HF_CONFIG_NOT_FOUND",
                f"Could not automatically fetch config for '{model}' from HuggingFace. VRAM calculations may be completely inaccurate unless you provide model_params_b.",
            )
        )

    mem_per_gpu = infer_gpu_memory_gb(gpu)
    param_b = (
        model_params_b_override
        if model_params_b_override is not None
        else parse_param_count_billions(model, hf_config)
    )
    weights_gb_total = estimate_weight_memory_gb(param_b, dtype, quantization)

    if mem_per_gpu is not None and weights_gb_total is not None:
        weights_per_gpu = weights_gb_total / max(tp * pp, 1)
        usable_vram_per_gpu = mem_per_gpu * 0.90
        kv_cache_budget = usable_vram_per_gpu - weights_per_gpu

        if weights_per_gpu > mem_per_gpu:
            issues.append(
                ValidationIssue(
                    "error",
                    "WEIGHTS_DO_NOT_FIT",
                    f"Estimated weights per GPU ({weights_per_gpu:.1f} GB) exceeds total GPU memory ({mem_per_gpu} GB).",
                )
            )
        elif kv_cache_budget <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "VRAM_EXHAUSTED",
                    f"Weights ({weights_per_gpu:.1f} GB) exceed the 90% utilization threshold ({usable_vram_per_gpu:.1f} GB). No room for KV cache.",
                )
            )
        elif kv_cache_budget < 8.0:
            issues.append(
                ValidationIssue(
                    "warning",
                    "CRITICAL_KV_CACHE_SHORTAGE",
                    f"Only ~{kv_cache_budget:.1f} GB left for KV cache/activations per GPU. High concurrency will cause OOM or severe queuing delays.",
                )
            )
        elif weights_per_gpu > (mem_per_gpu * 0.85):
            issues.append(
                ValidationIssue(
                    "warning",
                    "WEIGHTS_HEADROOM_LOW",
                    f"Estimated weights per GPU ({weights_per_gpu:.1f} GB) leaves low headroom on {mem_per_gpu} GB GPUs.",
                )
            )

        # Check for extreme underutilization / excessive scaling
        # If the model uses less than 10% of the VRAM on the GPU, scaling it across multiple GPUs via TP or PP is highly inefficient.
        if (tp * pp) > 1 and weights_per_gpu < (mem_per_gpu * 0.10):
            issues.append(
                ValidationIssue(
                    "warning",
                    "EXCESSIVE_GPU_SCALING",
                    f"Model weights per GPU ({weights_per_gpu:.1f} GB) utilize <10% of the available {mem_per_gpu} GB. Sharding such a small model across {tp*pp} GPUs via TP/PP introduces communication overhead that will likely destroy performance. Consider reducing GPUs or using Data Parallelism (DP) instead.",
                )
            )

        qnorm = normalize_quantization(quantization)
        if (
            param_b
            and param_b >= 100
            and mem_per_gpu <= 48
            and num_gpus <= 1
            and qnorm not in ("int4",)
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "MODEL_TOO_LARGE_FOR_SINGLE_GPU_CLASS",
                    f"{model} on {num_gpus}x{gpu} is very likely infeasible without aggressive quantization/offload.",
                )
            )

    else:
        issues.append(
            ValidationIssue(
                "info",
                "MEMORY_ESTIMATE_SKIPPED",
                "Could not estimate memory feasibility automatically. Use --model-params-b for a better check.",
            )
        )

    total_ctx = input_len + output_len

    if hf_max_ctx and total_ctx > hf_max_ctx:
        issues.append(
            ValidationIssue(
                "error",
                "CONTEXT_EXCEEDS_MODEL_CAPACITY",
                f"Requested input+output ({total_ctx}) exceeds the model's native maximum context window ({hf_max_ctx}).",
            )
        )

    if max_model_len >= 32768 and (mem_per_gpu or 0) <= 48:
        issues.append(
            ValidationIssue(
                "warning",
                "LONG_CONTEXT_ON_LOW_VRAM",
                f"max_model_len={max_model_len} on {gpu} may sharply limit concurrency due to KV cache pressure.",
            )
        )

    if quantization and normalize_quantization(quantization) in ("int4", "int8"):
        issues.append(
            ValidationIssue(
                "info",
                "QUANTIZATION_QUALITY_REMINDER",
                "Quantized models can materially change quality/latency behavior; validate outputs on your task before adopting in production.",
            )
        )

    return issues


# ============================================================
# Candidate generation
# ============================================================


def parse_vllm_version(ver_str: Optional[str]) -> Tuple[int, ...]:
    """
    Parses a vLLM version string into a tuple of integers for comparison.

    Args:
        ver_str: The version string (e.g., '0.11.2', 'v0.10.0').

    Returns:
        A tuple of parsed integer components.
    """
    if not ver_str:
        return (0, 0, 0)
    parts = []
    for p in ver_str.lower().replace("v", "").split("."):
        try:
            parts.append(int(re.sub(r"\D", "", p)))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def build_candidate_config(
    *,
    candidate_name: str,
    model: str,
    family: str,
    gpu: str,
    num_gpus: int,
    num_nodes: int,
    input_len: int,
    output_len: int,
    constraints: Optional[str],
    estimate_metrics: Dict[str, float],
    chat_template: Optional[str],
    dtype: str,
    quantization: Optional[str],
    trust_remote_code: bool,
    max_model_len_override: Optional[int],
    include_cuda_graph_sizes: bool,
    expect_shared_prefix: bool,
    prefer_streaming_smoothness: bool,
    enable_expert_parallel_override: Optional[bool],
    hf_config: Optional[Dict],
    model_params_b: Optional[float] = None,
    vllm_version_hint: str = "0.13.0",
) -> CandidateConfig:
    """
    Constructs a complete vLLM candidate configuration profile, detailing CLI arguments,
    the rationale behind them, and providing a test command using guidellm.

    Args:
        candidate_name: Target profile name ('latency', 'throughput', or 'balanced').
        model: Model identifier.
        family: Detected model family.
        gpu: GPU type.
        num_gpus: Total number of GPUs available.
        num_nodes: Total number of compute nodes.
        input_len: Expected average input sequence length.
        output_len: Expected average output sequence length.
        constraints: User-provided latency constraints.
        estimate_metrics: Parsed performance metrics from the llm-optimizer.
        chat_template: Optional path to a Jinja chat template.
        dtype: Preferred model data type.
        quantization: Optional quantization scheme.
        trust_remote_code: Whether to allow remote code execution from HF Hub.
        max_model_len_override: Optional explicit override for max_model_len.
        include_cuda_graph_sizes: Flag to inject small CUDA graph sizes.
        expect_shared_prefix: Flag indicating high shared prefix rates.
        prefer_streaming_smoothness: Flag to optimize for smooth token delivery.
        enable_expert_parallel_override: Force enable/disable MoE expert parallelism.
        hf_config: The Hugging Face config payload.
        model_params_b: Override for the parameter count.

    Returns:
        A populated CandidateConfig instance representing the profile.
    """
    total_ctx = input_len + output_len
    fam_defaults = model_family_defaults(family)
    hf_max_ctx = get_hf_max_context(hf_config)

    tp, pp, dp = infer_tp_pp_dp(
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        model=model,
        gpu=gpu,
        dtype=dtype,
        quantization=quantization,
        candidate=candidate_name,
        hf_config=hf_config,
        model_params_b_override=model_params_b,
    )

    max_model_len = choose_max_model_len(
        max_model_len_override, input_len, output_len, hf_max_ctx
    )
    gpu_mem_util = choose_gpu_memory_utilization(gpu, total_ctx, candidate_name)

    max_num_seqs = choose_max_num_seqs(
        gpu=gpu,
        num_gpus=num_gpus,
        tp=tp,
        pp=pp,
        total_ctx=total_ctx,
        candidate=candidate_name,
        est=estimate_metrics,
        model=model,
        dtype=dtype,
        quantization=quantization,
        hf_config=hf_config,
        model_params_b_override=model_params_b,
    )

    max_num_batched_tokens = choose_max_num_batched_tokens(
        gpu, num_gpus, total_ctx, candidate_name
    )
    stream_interval = choose_stream_interval(
        candidate_name, prefer_streaming_smoothness
    )
    max_partial_prefills, max_long_partial_prefills = choose_prefill_partials(
        candidate_name
    )
    async_sched = choose_async_scheduling(
        candidate_name, parse_constraints_for_ttft(constraints)
    )

    if enable_expert_parallel_override is None:
        eep = (
            is_moe_model(model, hf_config)
            or fam_defaults["enable_expert_parallel_default"]
        )
    else:
        eep = enable_expert_parallel_override

    if family == "default":
        prefix_cache = bool(expect_shared_prefix)
    else:
        prefix_cache = bool(expect_shared_prefix or fam_defaults["prefix_caching_bias"])

    args: List[str] = []
    rationale: Dict[str, str] = {}

    if chat_template:
        args += ["--chat-template", chat_template]
        rationale["--chat-template"] = (
            "Included because a chat template path was provided."
        )

    if tp > 1:
        args += ["--tensor-parallel-size", str(tp)]
        rationale["--tensor-parallel-size"] = (
            f"Set TP={tp} to shard matrix math within the node."
        )
    if pp > 1:
        args += ["--pipeline-parallel-size", str(pp)]
        rationale["--pipeline-parallel-size"] = (
            f"Set PP={pp} to partition the model layers across multiple nodes/devices."
        )
    if dp > 1:
        args += ["--data-parallel-size", str(dp)]
        rationale["--data-parallel-size"] = (
            f"Set DP={dp} to maximize throughput with independent replicas."
        )

    if tp == 1 and pp == 1 and dp == 1:
        rationale["parallelism"] = (
            "TP/PP/DP omitted because all are 1 (single-GPU config)."
        )

    if eep is True:
        args += ["--enable-expert-parallel"]
        rationale["--enable-expert-parallel"] = (
            f"Enabled because MoE architecture was detected or forced via preset/override."
        )
    elif eep is False:
        rationale["--enable-expert-parallel"] = "Not enabled."

    args += ["--gpu-memory-utilization", f"{gpu_mem_util:.2f}"]
    rationale["--gpu-memory-utilization"] = (
        f"Set to {gpu_mem_util:.2f} for {candidate_name} profile on this GPU class."
    )

    args += ["--dtype", dtype]
    rationale["--dtype"] = f"Set dtype={dtype}."

    if quantization:
        rationale["quantization"] = (
            f"Quantization hint='{quantization}' used for feasibility estimates. "
            "No vLLM quantization flag emitted automatically."
        )

    if trust_remote_code:
        args += ["--trust-remote-code"]
        rationale["--trust-remote-code"] = (
            "Enabled per model family default or explicit CLI option."
        )

    args += ["--max-model-len", str(max_model_len)]
    rationale["--max-model-len"] = (
        f"Set to {max_model_len} based on requested context limits and model capacity."
    )

    args += ["--max-num-seqs", str(max_num_seqs)]
    rationale["--max-num-seqs"] = (
        f"{candidate_name} scheduler cap chosen as {max_num_seqs} to balance concurrency vs TTFT risk."
    )

    v = parse_vllm_version(vllm_version_hint)

    # Chunked prefill became default in v0.6.0. Before that, we must explicitly enable it.
    if v < (0, 6, 0):
        args += ["--enable-chunked-prefill"]
        rationale["--enable-chunked-prefill"] = (
            f"Enabled as a good default for mixed/long prompt handling (required for vLLM < 0.6.0)."
        )
    else:
        # It's default in >= 0.6.0, but we can still emit it for clarity or let it be implicit.
        # Let's keep it implicit for cleaner arguments if version is recent.
        pass

    args += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    rationale["--max-num-batched-tokens"] = (
        f"Set to {max_num_batched_tokens} as the {candidate_name} chunk size limit."
    )

    # Partial prefill limits were introduced around v0.8.0.
    if v >= (0, 8, 0):
        args += ["--max-num-partial-prefills", str(max_partial_prefills)]
        rationale["--max-num-partial-prefills"] = (
            f"Set to {max_partial_prefills} for {candidate_name} prefill scheduling."
        )

        args += ["--max-long-partial-prefills", str(max_long_partial_prefills)]
        rationale["--max-long-partial-prefills"] = (
            f"Set to {max_long_partial_prefills} to limit long-prefill dominance in {candidate_name} profile."
        )

    if prefix_cache:
        args += ["--enable-prefix-caching"]
        rationale["--enable-prefix-caching"] = (
            "Enabled because shared prefixes are expected and/or family preset recommends it."
        )
    else:
        rationale["--enable-prefix-caching"] = (
            "Not enabled by default for this profile/family without shared-prefix expectation."
        )

    # Async scheduling was added in v0.10.0
    if async_sched and v >= (0, 10, 0):
        args += ["--async-scheduling"]
        rationale["--async-scheduling"] = (
            f"Enabled for {candidate_name} profile to improve throughput/ITL."
        )
    elif async_sched:
        rationale["--async-scheduling"] = (
            f"Disabled because vLLM version ({vllm_version_hint}) is too old to support async-scheduling (requires >= 0.10.0)."
        )
    else:
        rationale["--async-scheduling"] = (
            f"Disabled for {candidate_name} profile to protect TTFT under strict latency goals."
        )

    # Stream interval was added in v0.11.0 (or very late v0.10.x, but explicitly verified in 0.11.2)
    if v >= (0, 11, 0):
        args += ["--stream-interval", str(stream_interval)]
        rationale["--stream-interval"] = (
            f"Set to {stream_interval} for {candidate_name} streaming overhead tradeoff."
        )
    else:
        rationale["--stream-interval"] = (
            f"Disabled because vLLM version ({vllm_version_hint}) is too old to support stream-interval (requires >= 0.11.0)."
        )

    args += ["--disable-log-requests"]
    rationale["--disable-log-requests"] = "Enabled to reduce request logging overhead."

    if include_cuda_graph_sizes:
        args += ["--cuda-graph-sizes", "1", "2", "4", "8", "16", "32", "64", "128"]
        rationale["--cuda-graph-sizes"] = (
            "Included small CUDA graph capture set (vLLM 0.10.1.x style)."
        )

    args = [a for a in args if a != "--enforce-eager"]

    tuning_knobs: List[str] = []
    rates_str = ""

    if candidate_name == "latency":
        tuning_knobs = [
            "`--max-num-seqs`: Decrease to further protect TTFT (Time To First Token), or increase slightly if you have VRAM to spare.",
            "`--max-num-batched-tokens`: Lower values force the scheduler to yield more often, improving streaming smoothness at the cost of overall throughput.",
        ]
        rates_str = "1,2,4,8,16"
        rate_type = "concurrency"
    elif candidate_name == "throughput":
        tuning_knobs = [
            "`--kv-cache-dtype=fp8`: Highly recommended to reduce memory pressure and increase concurrent requests, assuming your hardware supports it (e.g., H100, L40S).",
            "`--max-num-seqs`: Increase to maximize batching efficiency, up to the limits of your KV cache.",
            "`--max-num-batched-tokens`: Increase to allow more prompt tokens to be processed in a single forward pass, heavily utilizing GPU compute.",
            "`--max-model-len`: If your workload doesn't need the full context window, decreasing this frees up significant KV cache for even higher concurrency.",
        ]
        rates_str = "2.0,4.0,8.0,16.0,32.0"
        rate_type = "throughput"
    else:
        tuning_knobs = [
            "`--max-num-seqs`: Tune up or down based on your SLA. Higher = better throughput, Lower = better latency.",
            "`--kv-cache-dtype=fp8`: Consider using fp8 KV cache to reclaim memory and boost concurrency if hardware supports it.",
            "`--max-model-len`: Lowering this limits maximum request size but allows more simultaneous requests.",
        ]
        if v < (0, 6, 0):
            tuning_knobs.append(
                "`--enable-chunked-prefill`: Toggling this off might improve small-prompt performance, but can severely penalize TTFT for concurrent requests if prompts are large."
            )
        rates_str = "1,8,32,64,128"
        rate_type = "concurrency"

    guidellm_cmd = (
        f"guidellm --target http://localhost:8000/v1 \\\n"
        f"  --model {model} \\\n"
        f"  --data-type emulated \\\n"
        f"  --emulated-prompt-tokens {input_len} \\\n"
        f"  --emulated-generated-tokens {output_len} \\\n"
        f"  --max-prompts 500 \\\n"
        f"  --rate-type {rate_type} \\\n"
        f"  --rate {rates_str}"
    )

    return CandidateConfig(
        name=candidate_name,
        args=args,
        rationale=rationale,
        tuning_knobs=tuning_knobs,
        guidellm_cmd=guidellm_cmd,
    )


# ============================================================
# Output helpers
# ============================================================


def print_issues(issues: List[ValidationIssue]) -> None:
    """
    Prints a formatted list of validation issues (errors, warnings, infos) to the console.

    Args:
        issues: A list of ValidationIssue instances.
    """
    print("\n=== Validation checks ===")
    if not issues:
        print("No obvious feasibility issues detected.")
        return
    for issue in issues:
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}.get(
            issue.level, issue.level.upper()
        )
        print(f"[{prefix}] {issue.code}: {issue.message}")


def print_candidate(candidate: CandidateConfig) -> None:
    """
    Displays a candidate configuration profile in a human-readable format,
    including Bash and OpenShift YAML formats, plus its rationale.

    Args:
        candidate: The CandidateConfig instance to display.
    """
    bash_args = format_args_multiline(candidate.args, " \\\n  ")
    yaml_args = format_args_multiline(candidate.args, "\n      ")

    print(f"\n--- Candidate: {candidate.name} ---")
    print("VLLM_ADDITIONAL_ARGS (Bash):")
    print(bash_args)

    print("\nOpenShift env snippet:")
    print("env:")
    print("  - name: VLLM_ADDITIONAL_ARGS")
    print("    value: >-")
    print(f"      {yaml_args}")

    print("\nRationale:")
    for k, v in candidate.rationale.items():
        print(f"  {k}: {v}")


def write_json_report(
    path: str,
    *,
    inputs: Dict[str, object],
    estimate: EstimateResult,
    family: str,
    issues: List[ValidationIssue],
    candidates: List[CandidateConfig],
) -> None:
    """
    Dumps all input details, estimation results, validation checks, and generated candidates
    into a structured JSON file.

    Args:
        path: Destination file path.
        inputs: Original user input arguments.
        estimate: Results from llm-optimizer execution.
        family: Identified model family.
        issues: Any detected feasibility issues.
        candidates: A list of the generated candidate configurations.
    """
    payload = {
        "inputs": inputs,
        "model_family": family,
        "llm_optimizer": {
            "returncode": estimate.returncode,
            "parsed_metrics": estimate.parsed,
            "stdout": estimate.stdout,
            "stderr": estimate.stderr,
        },
        "validation_issues": [asdict(i) for i in issues],
        "candidates": [
            {
                "name": c.name,
                "vllm_additional_args_list": c.args,
                "vllm_additional_args_string": shell_join(c.args),
                "vllm_additional_args_bash": format_args_multiline(c.args, " \\\n  "),
                "vllm_additional_args_yaml": format_args_multiline(c.args, "\n      "),
                "rationale": c.rationale,
                "tuning_knobs": c.tuning_knobs,
                "guidellm_cmd": c.guidellm_cmd,
            }
            for c in candidates
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote JSON report: {path}")


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the argparse parser for the script's command-line interface.

    Returns:
        An argparse.ArgumentParser instance.
    """
    p = argparse.ArgumentParser(
        description="Generate 3 starting vLLM configs from llm-optimizer estimate + model presets (no benchmark)."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Total number of GPUs across all nodes.",
    )
    p.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of physical nodes. Defaults to 1.",
    )
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)

    p.add_argument(
        "--constraints",
        default=None,
        help='Optional llm-optimizer constraints (e.g. "ttft:p95<8s;itl:p95<30ms")',
    )
    p.add_argument("--target", default=None)
    p.add_argument("--estimate-extra-args", default=None)

    p.add_argument(
        "--model-family",
        default=None,
        choices=["gpt-oss", "llama", "qwen", "granite", "default"],
        help="Override auto-detected model family preset",
    )
    p.add_argument(
        "--model-params-b",
        type=float,
        default=None,
        help="Override model parameter count in billions for feasibility checks (e.g., 0.5, 8, 70, 120)",
    )
    p.add_argument(
        "--quantization",
        default=None,
        help="Quantization hint for feasibility checks (fp8, int8, int4, awq, gptq, bf16, fp16)",
    )

    p.add_argument("--chat-template", default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument(
        "--vllm-version-hint",
        default="0.13.0",
        help="Optional vLLM version hint. Defaults to 0.13.0.",
    )

    p.add_argument("--trust-remote-code", action="store_true", default=None)
    p.add_argument(
        "--no-trust-remote-code", dest="trust_remote_code", action="store_false"
    )
    p.add_argument("--enable-expert-parallel", action="store_true", default=None)
    p.add_argument(
        "--disable-expert-parallel", dest="enable_expert_parallel", action="store_false"
    )
    p.add_argument("--expect-shared-prefix", action="store_true")
    p.add_argument("--prefer-streaming-smoothness", action="store_true")
    p.add_argument("--include-cuda-graph-sizes", action="store_true")

    p.add_argument("--fail-on-estimate-error", action="store_true")
    p.add_argument("--output-json", default=None)
    return p


def main() -> int:
    """
    Main entry point for the script. Orchestrates argument parsing, model family detection,
    llm-optimizer execution, configuration candidate generation, feasibility checks,
    and output rendering.

    Returns:
        Integer exit code (0 for success, non-zero for error/failure).
    """
    if not which("llm-optimizer"):
        print("ERROR: llm-optimizer not found in PATH", file=sys.stderr)
        return 2

    args = build_parser().parse_args()

    hf_config = fetch_hf_config(args.model)

    family = detect_model_family(args.model, args.model_family, hf_config)
    fam_defaults = model_family_defaults(family)

    if args.trust_remote_code is None:
        trust_remote_code = bool(fam_defaults["trust_remote_code_default"])
    else:
        trust_remote_code = bool(args.trust_remote_code)

    est_inp = EstimateInputs(
        model=args.model,
        gpu=args.gpu,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        constraints=args.constraints,
        target=args.target,
        extra_args=shlex.split(args.estimate_extra_args)
        if args.estimate_extra_args
        else [],
    )

    est = run_estimate(est_inp)
    if est.returncode != 0 and args.fail_on_estimate_error:
        print(est.stdout, file=sys.stdout)
        print(est.stderr, file=sys.stderr)
        return est.returncode

    candidates: List[CandidateConfig] = []

    # Auto-detect quantization if not explicitly provided
    detected_quant = detect_quantization_from_config(hf_config)
    effective_quantization = args.quantization or detected_quant

    for name in ("latency", "balanced", "throughput"):
        c = build_candidate_config(
            candidate_name=name,
            model=args.model,
            family=family,
            gpu=args.gpu,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            constraints=args.constraints,
            estimate_metrics=est.parsed,
            chat_template=args.chat_template,
            dtype=args.dtype,
            quantization=effective_quantization,
            trust_remote_code=trust_remote_code,
            max_model_len_override=args.max_model_len,
            include_cuda_graph_sizes=args.include_cuda_graph_sizes,
            expect_shared_prefix=args.expect_shared_prefix,
            prefer_streaming_smoothness=args.prefer_streaming_smoothness,
            enable_expert_parallel_override=args.enable_expert_parallel,
            hf_config=hf_config,
            model_params_b=args.model_params_b,
        )
        candidates.append(c)

    balanced = next(c for c in candidates if c.name == "balanced")

    def arg_value(
        a: List[str], flag: str, default: Optional[str] = None
    ) -> Optional[str]:
        try:
            i = a.index(flag)
            return a[i + 1]
        except Exception:
            return default

    tp_str = arg_value(balanced.args, "--tensor-parallel-size", "1")
    pp_str = arg_value(balanced.args, "--pipeline-parallel-size", "1")
    max_model_len_str = arg_value(
        balanced.args, "--max-model-len", str(args.max_model_len or 4096)
    )

    # We need to validate each candidate configuration individually because
    # extracting the max(TP) and max(PP) globally can create impossible
    # topologies (e.g., max_tp=4 from Latency + max_pp=2 from Balanced = 8 GPUs needed, but only 4 exist).
    issues_set = set()
    issues = []

    valid_vllm_args = (
        fetch_valid_vllm_args(args.vllm_version_hint)
        if args.vllm_version_hint
        else None
    )

    for c in candidates:
        c_tp = int(arg_value(c.args, "--tensor-parallel-size", "1") or "1")
        c_pp = int(arg_value(c.args, "--pipeline-parallel-size", "1") or "1")
        c_max_model_len = int(
            arg_value(c.args, "--max-model-len", str(args.max_model_len or 4096))
            or "4096"
        )

        quantization = args.quantization or detect_quantization_from_config(hf_config)

        c_issues = validate_feasibility(
            model=args.model,
            family=family,
            gpu=args.gpu,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            dtype=args.dtype,
            quantization=quantization,
            tp=c_tp,
            pp=c_pp,
            max_model_len=c_max_model_len,
            hf_config=hf_config,
            hf_max_ctx=get_hf_max_context(hf_config),
            model_params_b_override=args.model_params_b,
            candidate_args=c.args,
            valid_vllm_args=valid_vllm_args,
        )

        # Deduplicate issues
        for issue in c_issues:
            issue_tuple = (issue.level, issue.code, issue.message)
            if issue_tuple not in issues_set:
                issues_set.add(issue_tuple)
                issues.append(issue)

    print("=== llm-optimizer estimate command ===")
    print(shell_join(build_estimate_cmd(est_inp)))
    print(f"\nModel family preset: {family}")
    print(f"llm-optimizer returncode: {est.returncode}")
    if args.vllm_version_hint:
        print(f"vLLM version hint: {args.vllm_version_hint}")

    if est.parsed:
        print("\nParsed estimate metrics (best-effort):")
        for k, v in est.parsed.items():
            print(f"  {k}: {v}")
    else:
        print("\nParsed estimate metrics: none (parser did not match output format)")

    print_issues(issues)

    has_errors = any(i.level == "error" for i in issues)
    if has_errors:
        print(
            "\n*** WARNING: Fatal feasibility errors detected. Candidate configs below are for reference and likely won't run as-is. ***"
        )

    for c in candidates:
        print_candidate(c)

    print("\n=== Raw llm-optimizer stdout ===")
    print(est.stdout.strip() or "(empty)")
    if est.stderr.strip():
        print("\n=== Raw llm-optimizer stderr ===")
        print(est.stderr.strip())

    if args.output_json:
        write_json_report(
            args.output_json,
            inputs={
                "model": args.model,
                "gpu": args.gpu,
                "num_gpus": args.num_gpus,
                "num_nodes": args.num_nodes,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "constraints": args.constraints,
                "target": args.target,
                "model_params_b": args.model_params_b,
                "quantization": normalize_quantization(args.quantization),
                "vllm_version_hint": args.vllm_version_hint,
            },
            estimate=est,
            family=family,
            issues=issues,
            candidates=candidates,
        )

    return 3 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from typing import Dict, List, Optional, Tuple


# ============================================================
# Utilities & Hugging Face Integration
# ============================================================


def which(cmd: str) -> Optional[str]:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(d) / cmd
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def run_cmd(
    cmd: List[str], timeout: Optional[int] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, text=True, capture_output=True, timeout=timeout, check=False
    )


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def format_args_multiline(args: List[str], joiner: str) -> str:
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
    if not config:
        return None
    for key in ["max_position_embeddings", "max_sequence_length", "seq_length"]:
        if key in config and isinstance(config[key], int):
            return config[key]
    return None


def normalize_gpu_name(gpu: str) -> str:
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
    returncode: int
    stdout: str
    stderr: str
    parsed: Dict[str, float]


def build_estimate_cmd(inp: EstimateInputs) -> List[str]:
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
    name: str
    args: List[str]
    rationale: Dict[str, str]
    tuning_knobs: List[str]
    guidellm_cmd: str


def detect_model_family(
    model: str, override: Optional[str], config: Optional[Dict]
) -> str:
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
    """Calculates optimal 3D Parallelism (TP, PP, DP) constrained by cluster topology."""
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
    mem = infer_gpu_memory_gb(gpu)
    if mem is None:
        base = 0.90
    elif mem >= 80:
        base = 0.90
    elif mem >= 48:
        base = 0.88
    else:
        base = 0.85

    if total_ctx > 16000:
        base -= 0.02
    elif total_ctx > 8000:
        base -= 0.01

    if candidate == "latency":
        base -= 0.01
    elif candidate == "throughput":
        base += 0.01

    return max(0.82, min(0.93, round(base, 2)))


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
    if prefer_streaming_smoothness:
        return 1
    return {"latency": 1, "balanced": 5, "throughput": 10}[candidate]


def choose_prefill_partials(candidate: str) -> Tuple[int, int]:
    return {
        "latency": (2, 1),
        "balanced": (4, 1),
        "throughput": (8, 2),
    }[candidate]


def choose_async_scheduling(candidate: str, strict_ttft: bool) -> bool:
    if strict_ttft and candidate == "latency":
        return False
    return candidate in ("balanced", "throughput")


def choose_max_model_len(
    user_value: Optional[int],
    input_len: int,
    output_len: int,
    hf_max_ctx: Optional[int],
) -> int:
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
    return bool(constraints and "ttft" in constraints.lower())


# ============================================================
# Validation / feasibility checks
# ============================================================


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str


def normalize_quantization(q: Optional[str]) -> Optional[str]:
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
        "int8": "int8",
        "8bit": "int8",
        "int4": "int4",
        "4bit": "int4",
        "awq": "int4",
        "gptq": "int4",
    }
    return aliases.get(s, s)


def effective_bytes_per_param(dtype: str, quantization: Optional[str]) -> float:
    q = normalize_quantization(quantization)
    if q == "int4":
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
    if param_b is None:
        return None
    bpp = effective_bytes_per_param(dtype, quantization)
    overhead_factor = 1.15
    return param_b * bpp * overhead_factor


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
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    if num_gpus < 1:
        issues.append(
            ValidationIssue("error", "NUM_GPUS_INVALID", "num_gpus must be >= 1")
        )
        return issues

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
) -> CandidateConfig:
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

    args += ["--enable-chunked-prefill"]
    rationale["--enable-chunked-prefill"] = (
        "Enabled as a good default for mixed/long prompt handling."
    )

    args += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    rationale["--max-num-batched-tokens"] = (
        f"Set to {max_num_batched_tokens} as the {candidate_name} chunk size limit."
    )

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

    if async_sched:
        args += ["--async-scheduling"]
        rationale["--async-scheduling"] = (
            f"Enabled for {candidate_name} profile to improve throughput/ITL."
        )
    else:
        rationale["--async-scheduling"] = (
            f"Disabled for {candidate_name} profile to protect TTFT under strict latency goals."
        )

    args += ["--stream-interval", str(stream_interval)]
    rationale["--stream-interval"] = (
        f"Set to {stream_interval} for {candidate_name} streaming overhead tradeoff."
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
            "`--max-num-seqs`: Increase to maximize batching efficiency, up to the limits of your KV cache.",
            "`--max-num-batched-tokens`: Increase to allow more prompt tokens to be processed in a single forward pass, heavily utilizing GPU compute.",
            "`--max-model-len`: If your workload doesn't need the full context window, decreasing this frees up significant KV cache for even higher concurrency.",
        ]
        rates_str = "2.0,4.0,8.0,16.0,32.0"
        rate_type = "throughput"
    else:
        tuning_knobs = [
            "`--max-num-seqs`: Tune up or down based on your SLA. Higher = better throughput, Lower = better latency.",
            "`--max-model-len`: Lowering this limits maximum request size but allows more simultaneous requests.",
            "`--enable-chunked-prefill`: Toggling this off might improve small-prompt performance, but can severely penalize TTFT for concurrent requests if prompts are large.",
        ]
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
        "--vllm-version-hint", default=None, help="Optional vLLM version hint."
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
            quantization=args.quantization,
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

    for c in candidates:
        c_tp = int(arg_value(c.args, "--tensor-parallel-size", "1") or "1")
        c_pp = int(arg_value(c.args, "--pipeline-parallel-size", "1") or "1")
        c_max_model_len = int(
            arg_value(c.args, "--max-model-len", str(args.max_model_len or 4096))
            or "4096"
        )

        c_issues = validate_feasibility(
            model=args.model,
            family=family,
            gpu=args.gpu,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            dtype=args.dtype,
            quantization=args.quantization,
            tp=c_tp,
            pp=c_pp,
            max_model_len=c_max_model_len,
            hf_config=hf_config,
            hf_max_ctx=get_hf_max_context(hf_config),
            model_params_b_override=args.model_params_b,
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

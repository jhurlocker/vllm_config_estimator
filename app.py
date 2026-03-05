import os
import sys
import subprocess

# Add src to python path for local development
sys.path.insert(0, os.path.abspath("src"))

import tempfile
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/model_config", methods=["GET"])
def model_config():
    """
    Endpoint to fetch model quantization details from Hugging Face.
    Uses llm_optimizer's robust inference to handle standard configs, native Mistral formats, and metadata tags.
    """
    model_id = request.args.get("model")
    if not model_id:
        return jsonify({"error": "No model specified"}), 400

    try:
        from llm_optimizer.common import (
            get_quantization_from_hub,
            infer_precision_from_config,
        )
        from huggingface_hub import hf_hub_download

        # 1. Try to get precision from Hub metadata first (most reliable for Native Mistral and non-standard models)
        precision = get_quantization_from_hub(model_id)

        # 2. If not found, download config.json and infer from it
        if not precision:
            try:
                config_path = hf_hub_download(repo_id=model_id, filename="config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                precision = infer_precision_from_config(config, model_id)
            except Exception:
                # If config.json fails, precision remains None
                pass

        # Only return explicitly quantified formats to UI, otherwise return empty so UI snaps to "Native/Auto"
        if precision in ["fp4", "fp8", "int4", "int8"]:
            return jsonify({"quantization": precision})
        else:
            return jsonify({"quantization": ""})

    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/model_config", methods=["GET"])
def estimate():
    data = request.form

    # Required arguments
    cmd = [
        "python3",
        "vllm_start_config_from_estimate.py",
        "--model",
        data.get("model", ""),
        "--gpu",
        data.get("gpu", ""),
        "--num-gpus",
        data.get("num_gpus", ""),
        "--input-len",
        data.get("input_len", ""),
        "--output-len",
        data.get("output_len", ""),
    ]

    if data.get("num_nodes"):
        cmd.extend(["--num-nodes", str(data.get("num_nodes"))])

    # Optional arguments
    if data.get("constraints"):
        cmd.extend(["--constraints", str(data.get("constraints"))])
    if data.get("target"):
        cmd.extend(["--target", str(data.get("target"))])
    if data.get("model_family"):
        cmd.extend(["--model-family", str(data.get("model_family"))])
    if data.get("model_params_b"):
        cmd.extend(["--model-params-b", str(data.get("model_params_b"))])
    if data.get("quantization"):
        cmd.extend(["--quantization", str(data.get("quantization"))])
    if data.get("dtype") and data.get("dtype") != "auto":
        cmd.extend(["--dtype", str(data.get("dtype"))])
    if data.get("max_model_len"):
        cmd.extend(["--max-model-len", str(data.get("max_model_len"))])
    if data.get("vllm_version_hint"):
        cmd.extend(["--vllm-version-hint", str(data.get("vllm_version_hint"))])

    # Boolean arguments
    if "trust_remote_code" in data:
        cmd.append("--trust-remote-code")
    if "enable_expert_parallel" in data:
        cmd.append("--enable-expert-parallel")
    if "expect_shared_prefix" in data:
        cmd.append("--expect-shared-prefix")
    if "prefer_streaming_smoothness" in data:
        cmd.append("--prefer-streaming-smoothness")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        json_file = tf.name
        cmd.extend(["--output-json", json_file])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        output_json = None
        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    output_json = json.load(f)
            except Exception:
                pass

        response = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "json": output_json,
        }

        # Catch specific precision errors from stderr if the process failed but didn't crash hard
        if result.returncode != 0 and "precision not supported on" in result.stderr:
            # Create a synthetic JSON response with the error so the UI renders it cleanly
            response["json"] = {
                "candidates": [],
                "validation_issues": [
                    {
                        "level": "error",
                        "code": "PRECISION_NOT_SUPPORTED",
                        "message": result.stderr.strip().split("\n")[
                            -1
                        ],  # Get the last line
                    }
                ],
            }

    except subprocess.TimeoutExpired:
        response = {
            "stdout": "",
            "stderr": "Command timed out after 10 minutes",
            "returncode": 124,
            "json": None,
        }
    except Exception as e:
        response = {"stdout": "", "stderr": str(e), "returncode": 1, "json": None}
    finally:
        if os.path.exists(json_file):
            os.remove(json_file)

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

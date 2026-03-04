import os
import subprocess
import tempfile
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/estimate", methods=["POST"])
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

import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add project root to python path so we can import app.py
sys.path.insert(0, os.path.abspath("."))
# Add src to python path so app can import modules (it does this itself, but doing it here ensures mocks work)
sys.path.insert(0, os.path.abspath("src"))

# Import the flask app
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_precision_error_in_stderr(client):
    """
    Simulate llm-optimizer crashing and printing error to stderr (classic behavior).
    """
    with patch("subprocess.run") as mock_run:
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = (
            "Traceback...\nValueError: fp8 precision not supported on A100"
        )
        mock_run.return_value = mock_result

        # Call the endpoint
        resp = client.post(
            "/estimate",
            data={
                "model": "foo",
                "gpu": "A100",
                "num_gpus": 1,
                "input_len": 10,
                "output_len": 10,
                "quantization": "fp8",
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()

        # Check returncode is forced to 1
        assert data["returncode"] == 1

        # Check validation issues
        assert "json" in data
        assert "validation_issues" in data["json"]
        issues = data["json"]["validation_issues"]
        assert len(issues) > 0
        assert issues[0]["code"] == "PRECISION_NOT_SUPPORTED"
        assert "fp8 precision not supported" in issues[0]["message"]


def test_precision_error_in_json_output(client):
    """
    Simulate llm-optimizer catching the error internally and writing it to JSON report
    while exiting with 0 (the tricky case).
    """
    with (
        patch("subprocess.run") as mock_run,
        patch("tempfile.NamedTemporaryFile") as mock_temp,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open") as mock_open,
        patch("os.remove"),
    ):
        # Mock subprocess result (success exit code)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Log output..."
        mock_result.stderr = ""  # No stderr here
        mock_run.return_value = mock_result

        # Mock JSON file existence and content
        mock_exists.return_value = True

        # The JSON output written by the tool
        tool_output = {
            "llm_optimizer": {
                "returncode": 0,
                "stderr": "ValueError: fp8 precision not supported on A100",
                "stdout": "",
            },
            "candidates": [],
            "validation_issues": [],  # Tool didn't add it as a formal issue, just crashed
        }

        # Mock file reading
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(tool_output)
        mock_open.return_value = mock_file

        # Call endpoint
        resp = client.post(
            "/estimate", data={"model": "foo", "gpu": "A100", "quantization": "fp8"}
        )

        data = resp.get_json()

        # Should detect the error inside the JSON
        assert data["returncode"] == 1  # Forced failure
        issues = data["json"]["validation_issues"]
        assert len(issues) == 1
        assert issues[0]["code"] == "PRECISION_NOT_SUPPORTED"


def test_validation_warnings_passthrough(client):
    """
    Verify that existing validation warnings (e.g. Pipeline Parallel) are passed through
    unaffected when there are NO precision errors.
    """
    with (
        patch("subprocess.run") as mock_run,
        patch("tempfile.NamedTemporaryFile") as mock_temp,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open") as mock_open,
        patch("os.remove"),
    ):
        # Mock subprocess result (pure success)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        mock_exists.return_value = True

        # The JSON output with a warning
        tool_output = {
            "llm_optimizer": {"returncode": 0, "stderr": ""},
            "candidates": [{"name": "foo"}],
            "validation_issues": [
                {
                    "level": "warning",
                    "code": "MULTI_NODE_PIPELINE_PARALLEL",
                    "message": "Warning text",
                }
            ],
        }

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(tool_output)
        mock_open.return_value = mock_file

        resp = client.post("/estimate", data={"model": "foo"})
        data = resp.get_json()

        # Check returncode matches subprocess (0)
        assert data["returncode"] == 0

        # Check warnings are present
        assert "validation_issues" in data["json"]
        issues = data["json"]["validation_issues"]
        assert len(issues) == 1
        assert issues[0]["code"] == "MULTI_NODE_PIPELINE_PARALLEL"


def test_mixed_precision_error_and_warnings(client):
    """
    Verify that if we catch a precision error, we append it to existing warnings
    instead of replacing them.
    """
    with (
        patch("subprocess.run") as mock_run,
        patch("tempfile.NamedTemporaryFile") as mock_temp,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open") as mock_open,
        patch("os.remove"),
    ):
        # Case: Tool finds PP warning, AND crashes with precision error in stderr
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""  # FIX: Set string value so it is serializable
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        mock_exists.return_value = True

        tool_output = {
            "llm_optimizer": {
                "returncode": 0,
                "stderr": "ValueError: precision not supported on A100",
            },
            "validation_issues": [{"code": "EXISTING_WARNING", "message": "foo"}],
        }

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(tool_output)
        mock_open.return_value = mock_file

        resp = client.post("/estimate", data={"model": "foo"})
        data = resp.get_json()

        # Should be forced error
        assert data["returncode"] == 1

        # Should have BOTH issues
    issues = data["json"]["validation_issues"]
    codes = [i["code"] for i in issues]
    assert "EXISTING_WARNING" in codes
    assert "PRECISION_NOT_SUPPORTED" in codes


def test_invalid_gpu_warning(client):
    """
    Test that invalid GPU values result in a WARNING, not an error.
    """
    with (
        patch("subprocess.run") as mock_run,
        patch("tempfile.NamedTemporaryFile") as mock_temp,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open") as mock_open,
        patch("os.remove"),
    ):
        # Mock subprocess result with non-zero exit code but specific error msg
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stdout = ""
        mock_result.stderr = (
            "Error: Invalid value for '--gpu': 'L4' is not one of 'A100'..."
        )
        mock_run.return_value = mock_result

        # Mock JSON file existence (script produces it even on error)
        mock_exists.return_value = True

        tool_output = {
            "llm_optimizer": {"returncode": 2, "stderr": mock_result.stderr},
            "candidates": [],  # Empty candidates
            "validation_issues": [],
        }

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps(tool_output)
        mock_open.return_value = mock_file

        # Call the endpoint
        resp = client.post(
            "/estimate",
            data={
                "model": "foo",
                "gpu": "L4",
                "num_gpus": 1,
                "input_len": 10,
                "output_len": 10,
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()

        # Check returncode is forced to 0 (Success)
        assert data["returncode"] == 0

        # Check we have a warning
        assert "json" in data
        assert "validation_issues" in data["json"]
        issues = data["json"]["validation_issues"]
        assert len(issues) > 0
        assert issues[0]["code"] == "GPU_NOT_PROFILED"
        assert issues[0]["level"] == "warning"

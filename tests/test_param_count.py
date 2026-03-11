import json
import urllib.request
from vllm_start_config_from_estimate import parse_param_count_billions

url = "https://huggingface.co/RedHatAI/granite-4.0-h-tiny/raw/main/config.json"
req = urllib.request.Request(url)
with urllib.request.urlopen(req, timeout=5) as response:
    config = json.loads(response.read().decode("utf-8"))

param_b = parse_param_count_billions("RedHatAI/granite-4.0-h-tiny", config)
print(f"Estimated params: {param_b} billion")

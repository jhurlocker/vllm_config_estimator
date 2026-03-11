import json
import urllib.request
from llm_optimizer.common import calculate_model_parameters_from_config

url = "https://huggingface.co/RedHatAI/granite-4.0-h-tiny/raw/main/config.json"
req = urllib.request.Request(url)
with urllib.request.urlopen(req, timeout=5) as response:
    config = json.loads(response.read().decode("utf-8"))

params = calculate_model_parameters_from_config(config)
print(f"Calculated parameters: {params / 1e9:.3f} Billion")

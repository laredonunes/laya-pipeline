"""Professor direto no Amazon Bedrock (API Converse), com as credenciais AWS
do ambiente (perfil, variáveis ou role). Alternativa ao proxy quando a
rotulagem roda numa máquina com acesso à conta: não precisa do token do
proxy, mas o uso não aparece no /usage do proxy — por isso os tokens
consumidos ficam em runs/<versão>/teacher_usage.json."""

import threading
from typing import Dict, List, Optional

from .proxy import ProxyError

DEFAULT_MODEL = "deepseek.v3.2"
DEFAULT_REGION = "sa-east-1"

_clients: Dict[str, object] = {}
_lock = threading.Lock()
usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def _client(region: str):
    with _lock:
        if region not in _clients:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as error:
                raise ProxyError("provider bedrock precisa do boto3: pip install 'laya-pipeline[bedrock]'") from error
            _clients[region] = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=300))
        return _clients[region]


def chat(messages: List[Dict[str, str]], model: Optional[str] = None, region: Optional[str] = None,
         temperature: float = 0.2, max_tokens: int = 2048) -> str:
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    convo = [{"role": m["role"], "content": [{"text": m["content"]}]}
             for m in messages if m["role"] != "system"]
    try:
        resp = _client(region or DEFAULT_REGION).converse(
            modelId=model or DEFAULT_MODEL, system=system, messages=convo,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens})
    except ProxyError:
        raise
    except Exception as error:  # noqa: BLE001 — botocore tem muitas classes; o teacher repete
        raise ProxyError("bedrock: %s: %s" % (type(error).__name__, str(error)[:300])) from error
    with _lock:
        usage["calls"] += 1
        usage["input_tokens"] += resp.get("usage", {}).get("inputTokens", 0)
        usage["output_tokens"] += resp.get("usage", {}).get("outputTokens", 0)
    # Modelos com raciocínio devolvem blocos reasoningContent antes do texto.
    text = "".join(block.get("text", "") for block in resp["output"]["message"]["content"])
    if not text:
        raise ProxyError("bedrock: resposta sem texto (stopReason=%s)" % resp.get("stopReason"))
    return text

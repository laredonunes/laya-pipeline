"""Cliente mínimo (só stdlib) do hermes-usage-proxy do provedor-ia.

Todo acesso externo do pipeline passa pelo proxy: chat com o LLM professor
(rotulagem e geração de textos) e S3 (publicação do checkpoint via URL
pré-assinada). Assim o pipeline roda igual no notebook (Studio Lab/Kaggle),
na estação de trabalho ou num job na AWS, só com HERMES_PROXY_URL e
HERMES_TOKEN — sem credenciais AWS nem chaves de LLM."""

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

CHAT_PATHS = {
    "deepseek": ("/llm/chat/completions", "deepseek-chat"),
    "qwen": ("/qwen/chat/completions", "qwen3-coder"),
    "openrouter": ("/openrouter/api/v1/chat/completions", None),
}

RETRY_STATUS = {429, 500, 502, 503, 504}


class ProxyError(RuntimeError):
    pass


def _config():
    base = os.environ.get("HERMES_PROXY_URL", "").rstrip("/")
    token = os.environ.get("HERMES_TOKEN", "")
    if not base or not token:
        raise ProxyError("defina as variáveis de ambiente HERMES_PROXY_URL e HERMES_TOKEN")
    return base, token


def request_json(method: str, path: str, body: Optional[Dict[str, Any]] = None,
                 timeout: float = 180, retries: int = 3) -> Any:
    base, token = _config()
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"X-Hermes-Token": token}
    if data is not None:
        headers["Content-Type"] = "application/json"
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:1000]
            last = ProxyError("HTTP %d em %s %s: %s" % (error.code, method, path, detail))
            if error.code not in RETRY_STATUS:
                raise last
        except (urllib.error.URLError, TimeoutError) as error:
            last = ProxyError(
                "sem conexão com o proxy (%s). A instância pode estar desligada fora do "
                "expediente — ligue-a pelo provedor-ia (Lambda hermes-llm-proxy-control)." % error)
        if attempt < retries:
            time.sleep(2 ** attempt * 2)
    raise last


def chat(provider: str, messages: List[Dict[str, str]], model: Optional[str] = None,
         temperature: float = 0.2, json_mode: bool = True, max_tokens: int = 2048) -> str:
    """Devolve o texto da resposta do LLM (OpenAI-compatível)."""
    path, default_model = CHAT_PATHS[provider]
    body: Dict[str, Any] = {"messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens}
    if model or default_model:
        body["model"] = model or default_model
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = request_json("POST", path, body)
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProxyError("resposta de chat inesperada: %s" % json.dumps(resp)[:500])


def s3_upload(local_path: str, key: str) -> int:
    """Envia um arquivo direto ao S3 por URL pré-assinada (até 5 GB)."""
    content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    signed = request_json("POST", "/s3/presign",
                          {"key": key, "method": "PUT", "content_type": content_type})
    size = os.path.getsize(local_path)
    with open(local_path, "rb") as f:
        req = urllib.request.Request(
            signed["url"], data=f, method="PUT",
            headers={**signed.get("headers", {}), "Content-Length": str(size)})
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                resp.read()
        except urllib.error.HTTPError as error:
            raise ProxyError("HTTP %d no upload de %s: %s" % (
                error.code, key, error.read().decode("utf-8", "replace")[:500]))
    return size


def s3_download(key: str, local_path: str) -> int:
    signed = request_json("POST", "/s3/presign", {"key": key, "method": "GET"})
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with urllib.request.urlopen(signed["url"], timeout=3600) as resp, open(local_path, "wb") as f:
        size = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                return size
            f.write(chunk)
            size += len(chunk)

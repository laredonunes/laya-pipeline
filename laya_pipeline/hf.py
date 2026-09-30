"""Acesso ao checkpoint base no Hugging Face Hub."""

import json
import os
from typing import Optional


def fix_tokenizer_config(model_dir: str) -> None:
    """Mesma correção que `laya.agent._fix_tokenizer_config` aplica (laya 0.3.5,
    Apache-2.0): o tokenizer do mmBERT guarda `extra_special_tokens` como lista
    e algumas versões do transformers só aceitam mapeamento. Repetida aqui
    para que as etapas sem torch (rotulagem) também consigam carregar o
    tokenizer."""
    cfg_file = os.path.join(model_dir, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    with open(cfg_file) as f:
        tcfg = json.load(f)
    changed = False
    if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tcfg.pop("backend", None)
        tcfg.pop("is_local", None)
        changed = True
    extra = tcfg.get("extra_special_tokens")
    if isinstance(extra, list):
        tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
        changed = True
    if changed:
        # O arquivo do cache do HF é um symlink para o blob; grava uma cópia
        # real no lugar para não alterar o blob compartilhado.
        if os.path.islink(cfg_file):
            os.unlink(cfg_file)
        with open(cfg_file, "w") as f:
            json.dump(tcfg, f, indent=2)


def base_model_dir(model_id: str, subfolder: Optional[str] = None,
                   tokenizer_only: bool = False) -> str:
    """Baixa (ou reaproveita do cache) o checkpoint base e devolve o diretório
    com model.safetensors, encoder/, tokenizer/ e rl_agent_config.json."""
    if os.path.isdir(model_id):
        path = os.path.join(model_id, subfolder or "")
        fix_tokenizer_config(path)
        return path
    from huggingface_hub import snapshot_download

    prefix = "%s/" % subfolder if subfolder else ""
    patterns = [prefix + "tokenizer/*"] if tokenizer_only else [
        prefix + "*.json", prefix + "*.safetensors", prefix + "encoder/*", prefix + "tokenizer/*"]
    root = snapshot_download(model_id, allow_patterns=patterns)
    path = os.path.join(root, subfolder or "")
    fix_tokenizer_config(path)
    return path

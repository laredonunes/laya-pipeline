"""Publica o checkpoint treinado + relatórios no S3 (via proxy) e copia o
relatório para reports/ do repositório da tarefa, que vai para o git.

Layout no bucket (contrato com o provedor-ia, que monta a imagem de
inferência a partir daqui):
  models/laya/<tarefa>/<versão>/checkpoint/{model.safetensors, encoder/, tokenizer/, rl_agent_config.json}
  models/laya/<tarefa>/<versão>/{report.md, eval_baseline.json, eval_trained.json, task.yaml, manifest.json}"""

import hashlib
import json
import os
import shutil
import time
from typing import Dict, List

from . import __version__, proxy
from .evaluate import check_goals
from .task import Task


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(task: Task, force: bool = False, dry_run: bool = False) -> Dict[str, object]:
    ckpt = os.path.join(task.run_dir, "checkpoint")
    trained_path = os.path.join(task.run_dir, "eval_trained.json")
    if not os.path.exists(os.path.join(ckpt, "model.safetensors")):
        raise SystemExit("sem checkpoint em %s: rode `train` primeiro" % ckpt)
    if not os.path.exists(trained_path):
        raise SystemExit("checkpoint sem avaliação: rode `eval --trained` antes de publicar")
    with open(trained_path, encoding="utf-8") as f:
        trained = json.load(f)
    with open(os.path.join(ckpt, "rl_agent_config.json")) as f:
        if json.load(f).get("training", {}).get("smoke_test"):
            raise SystemExit("este checkpoint é de teste de fumaça (--max-steps); não publico")
    goals = check_goals(task, trained)
    if any(v is False for v in goals.values()) and not force:
        raise SystemExit("metas não atingidas %s; use --force para publicar mesmo assim" % goals)

    files: List[tuple] = []
    for dirpath, _, names in os.walk(ckpt):
        for name in names:
            local = os.path.join(dirpath, name)
            files.append((local, "checkpoint/" + os.path.relpath(local, ckpt).replace(os.sep, "/")))
    for name in ("report.md", "eval_baseline.json", "eval_trained.json", "train_log.jsonl"):
        local = os.path.join(task.run_dir, name)
        if os.path.exists(local):
            files.append((local, name))
    files.append((task.path("task.yaml"), "task.yaml"))

    manifest = {
        "task": task.name, "version": task.version, "pipeline_version": __version__,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": task.base_model, "context": task.context,
        "metrics": trained.get("overall"), "goals": goals,
        "files": {key: {"bytes": os.path.getsize(local), "sha256": _sha256(local)} for local, key in files},
    }
    manifest_path = os.path.join(task.run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    files.append((manifest_path, "manifest.json"))

    prefix = task.s3_key_prefix
    for local, key in files:
        if dry_run:
            print("[dry-run] %s → s3://…/%s%s" % (local, prefix, key))
        else:
            size = proxy.s3_upload(local, prefix + key)
            print("enviado %s%s (%.1f MB)" % (prefix, key, size / 1e6), flush=True)

    os.makedirs(task.path("reports"), exist_ok=True)
    shutil.copy(os.path.join(task.run_dir, "report.md"), task.path("reports", "%s.md" % task.version))
    shutil.copy(manifest_path, task.path("reports", "%s.manifest.json" % task.version))
    return {"s3_prefix": prefix, "files": len(files), "goals": goals}

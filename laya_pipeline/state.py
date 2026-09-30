"""Estado de uma rodada (runs/<versão>/) e os portões de aprovação humana.

Portões:
- data:   amostra rotulada revisada → libera `train`;
- report: relatório de avaliação revisado → libera `publish`.
Na fase de automação (Step Functions), cada portão vira uma pausa com
task token e um botão na página; o arquivo approvals.json continua sendo
a fonte da verdade para quem roda à mão."""

import json
import os
import time
from typing import Any, Dict, List

from .task import Task

GATES = ("data", "report")


def _path(task: Task) -> str:
    return os.path.join(task.run_dir, "approvals.json")


def approvals(task: Task) -> Dict[str, Any]:
    if not os.path.exists(_path(task)):
        return {}
    with open(_path(task), encoding="utf-8") as f:
        return json.load(f)


def approve(task: Task, gate: str, by: str, note: str = "") -> None:
    if gate not in GATES:
        raise SystemExit("portão desconhecido: %s (use %s)" % (gate, ", ".join(GATES)))
    os.makedirs(task.run_dir, exist_ok=True)
    data = approvals(task)
    data[gate] = {"by": by, "note": note, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(_path(task), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def require(task: Task, gate: str) -> None:
    if gate not in approvals(task):
        raise SystemExit(
            "portão '%s' ainda não aprovado para %s %s. Revise %s e rode "
            "`laya-pipeline approve %s --by <nome>`." % (
                gate, task.name, task.version,
                "runs/%s/sample.md" % task.version if gate == "data" else "runs/%s/report.md" % task.version,
                gate))


def status(task: Task) -> List[Dict[str, Any]]:
    """Etapas na ordem, com feito/pendente — responde "qual o próximo passo?"."""
    run = task.run_dir
    exists = lambda name: os.path.exists(os.path.join(run, name))  # noqa: E731
    texts = task.path(task.data["texts"])
    appr = approvals(task)
    return [
        {"step": "texts", "done": os.path.exists(texts) and os.path.getsize(texts) > 0},
        {"step": "label", "done": exists("labeled.jsonl")},
        {"step": "split", "done": exists("train.jsonl") and exists("eval.jsonl")},
        {"step": "sample", "done": exists("sample.md")},
        {"step": "approve data", "done": "data" in appr},
        {"step": "eval baseline", "done": exists("eval_baseline.json")},
        {"step": "train", "done": exists("checkpoint/model.safetensors")},
        {"step": "eval trained", "done": exists("eval_trained.json")},
        {"step": "approve report", "done": "report" in appr},
        {"step": "publish", "done": exists("manifest.json")
         and os.path.exists(task.path("reports", "%s.md" % task.version))},
    ]

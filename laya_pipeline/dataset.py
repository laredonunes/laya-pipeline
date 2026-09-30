"""Divisão treino/avaliação, estatísticas e amostra para revisão humana."""

import hashlib
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

from .task import Task, option_keys
from .teacher import read_jsonl


def _bucket(case_id: str) -> float:
    """Posição estável em [0, 1) derivada do id: o mesmo texto cai sempre do
    mesmo lado da divisão, mesmo se o dataset crescer ou for rotulado de novo.
    Isso impede que um texto de avaliação vaze para o treino numa versão futura."""
    return int(hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:8], 16) / 16 ** 8


def split(task: Task) -> Tuple[int, int]:
    rows = read_jsonl(os.path.join(task.run_dir, "labeled.jsonl"))
    if not rows:
        raise SystemExit("nada rotulado ainda: rode `label` primeiro")
    frac = float(task.data["eval_fraction"])
    train = [r for r in rows if _bucket(r["id"]) >= frac]
    evaluation = [r for r in rows if _bucket(r["id"]) < frac]
    for name, part in (("train.jsonl", train), ("eval.jsonl", evaluation)):
        with open(os.path.join(task.run_dir, name), "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(train), len(evaluation)


def winner(probabilities: Dict[str, float]) -> str:
    return max(probabilities, key=probabilities.get)


def stats(task: Task, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribuição do rótulo vencedor por pergunta e confiança média do
    professor. Classe com quase nenhum exemplo = o modelo não vai aprendê-la."""
    out: Dict[str, Any] = {"cases": len(rows), "questions": {}}
    for qid, q in task.questions.items():
        counts = Counter(winner(r["gold"][qid]["probabilities"]) for r in rows if qid in r["gold"])
        top = [max(r["gold"][qid]["probabilities"].values()) for r in rows if qid in r["gold"]]
        out["questions"][qid] = {
            "distribution": {k: counts.get(k, 0) for k in option_keys(q)},
            "teacher_mean_top_prob": round(sum(top) / len(top), 3) if top else None,
        }
    tokens = [r["state_tokens"] for r in rows if r.get("state_tokens")]
    if tokens:
        tokens.sort()
        budget = task.context["max_len"] - task.context["head_max_len"] - 8
        out["state_tokens"] = {"p50": tokens[len(tokens) // 2], "max": tokens[-1],
                               "truncated": sum(1 for t in tokens if t >= budget)}
    return out


def sample(task: Task, n: int = 20, seed: int = 0) -> str:
    """Gera runs/<versão>/sample.md: n casos com o texto e o que o professor
    respondeu, para a revisão humana antes do treino."""
    rows = read_jsonl(os.path.join(task.run_dir, "labeled.jsonl"))
    if not rows:
        raise SystemExit("nada rotulado ainda: rode `label` primeiro")
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(n, len(rows)))
    st = stats(task, rows)
    lines = ["# Amostra para revisão — %s %s" % (task.name, task.version), "",
             "Total rotulado: %d casos." % st["cases"], "", "## Distribuição dos rótulos", ""]
    for qid, info in st["questions"].items():
        lines.append("- **%s**: %s (confiança média do professor: %s)" % (
            qid, ", ".join("%s=%d" % kv for kv in info["distribution"].items()),
            info["teacher_mean_top_prob"]))
    lines += ["", "## Casos", ""]
    for i, row in enumerate(chosen, 1):
        text = row["state"]
        if len(text) > 1500:
            text = text[:1500] + " […]"
        lines += ["### %d. `%s`" % (i, row["id"]), "", "> " + text.replace("\n", "\n> "), ""]
        for qid in task.questions:
            probs = row["gold"][qid]["probabilities"]
            lines.append("- **%s** → `%s` %s" % (qid, winner(probs), json.dumps(probs, ensure_ascii=False)))
        lines += ["- revisão: [ ] correto  [ ] errado — comentário:", ""]
    path = os.path.join(task.run_dir, "sample.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

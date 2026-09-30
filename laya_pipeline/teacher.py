"""O "professor": um LLM (via proxy) que gera textos sintéticos e rotula cada
texto com uma distribuição de probabilidade por pergunta.

O treino RLCD do Laya aprende a imitar essas distribuições (não só o rótulo
vencedor), então a qualidade do professor é o teto do modelo — por isso a
amostra rotulada passa por revisão humana antes de gastar GPU."""

import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import proxy
from .task import Task, option_keys


def text_id(state: str) -> str:
    return hashlib.sha1(state.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: str, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_prompt(task: Task, key: str) -> str:
    with open(task.path(task.teacher[key]), encoding="utf-8") as f:
        return f.read().strip()


def render_questions(task: Task) -> str:
    """Descreve as perguntas e as chaves exatas que o JSON de resposta deve ter."""
    lines = []
    for qid, q in task.questions.items():
        lines.append("### %s (%s)" % (qid, q["type"]))
        lines.append(q["instructions"])
        crit = q.get("criteria")
        if q["type"] == "choice":
            for key, desc in crit.items():
                lines.append('- "%s"%s' % (key, ": %s" % desc if desc else ""))
        elif q["type"] == "score":
            for i, desc in enumerate(crit):
                lines.append('- "%d": %s' % (i, desc))
        else:
            crit = crit or {}
            lines.append('- "true": %s' % (crit.get("true") or "sim, a afirmação vale"))
            lines.append('- "false": %s' % (crit.get("false") or "não, a afirmação não vale"))
        lines.append("")
    return "\n".join(lines)


def output_schema(task: Task) -> str:
    example = {qid: {k: "<prob>" for k in option_keys(q)} for qid, q in task.questions.items()}
    return json.dumps(example, ensure_ascii=False)


def parse_gold(task: Task, content: str) -> Dict[str, Dict[str, Any]]:
    """Valida e normaliza a resposta do professor. Levanta ValueError se algo
    faltar — a chamada é repetida, em vez de gravar um rótulo pela metade."""
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError("resposta sem JSON")
    data = json.loads(match.group(0))
    gold = {}
    for qid, q in task.questions.items():
        probs = data.get(qid)
        if not isinstance(probs, dict):
            raise ValueError("pergunta %s ausente" % qid)
        keys = option_keys(q)
        values = []
        for key in keys:
            value = probs.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError("%s.%s: probabilidade inválida %r" % (qid, key, value))
            values.append(float(value))
        total = sum(values)
        if total <= 0:
            raise ValueError("%s: probabilidades somam zero" % qid)
        gold[qid] = {"probabilities": {k: round(v / total, 4) for k, v in zip(keys, values)}}
    return gold


class StateTruncator:
    """Corta o texto no mesmo ponto em que o Laya cortaria, para que o
    professor rotule exatamente o que o modelo vai ver. Usa só o tokenizer
    (sem torch). Se o tokenizer não estiver disponível, não corta."""

    def __init__(self, task: Task):
        self.budget = task.context["max_len"] - task.context["head_max_len"] - 8
        self.tok = None
        try:
            from transformers import AutoTokenizer

            from .hf import base_model_dir

            model_dir = base_model_dir(task.base_model["id"], task.base_model.get("subfolder"),
                                       tokenizer_only=True)
            self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        except Exception as error:  # noqa: BLE001 — degrada para "sem corte"
            print("aviso: tokenizer indisponível (%s); textos não serão truncados" % error)

    def __call__(self, state: str) -> Tuple[str, Optional[int]]:
        if self.tok is None:
            return state, None
        ids = self.tok(state, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.budget:
            return state, len(ids)
        return self.tok.decode(ids[: self.budget]), self.budget


def label(task: Task, limit: Optional[int] = None) -> Dict[str, int]:
    """Rotula data/texts.jsonl → runs/<versão>/labeled.jsonl. Retomável: textos
    já rotulados são pulados, então dá para interromper e continuar."""
    texts = read_jsonl(task.path(task.data["texts"]))
    if not texts:
        raise SystemExit("nenhum texto em %s (rode `ingest` depois de colocar arquivos em data/documentos, ou use gen-texts)" % task.data["texts"])
    os.makedirs(task.run_dir, exist_ok=True)
    out_path = os.path.join(task.run_dir, "labeled.jsonl")
    fail_path = os.path.join(task.run_dir, "label_failures.jsonl")
    done = {row["id"] for row in read_jsonl(out_path)}
    todo = []
    for row in texts:
        row.setdefault("id", text_id(row["state"]))
        if row["id"] not in done:
            todo.append(row)
    if limit is not None:
        todo = todo[:limit]

    truncate = StateTruncator(task)
    system = _read_prompt(task, "prompt")
    questions_block = render_questions(task)
    schema = output_schema(task)
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "skipped": len(done)}

    def work(row):
        state, n_tokens = truncate(row["state"])
        user = (
            "## Perguntas\n\n%s\n## Texto\n\n%s\n\n## Resposta\n\nResponda APENAS com um JSON "
            "neste formato, onde cada <prob> é sua probabilidade (0 a 1) de que aquela opção "
            "seja a correta; as probabilidades de cada pergunta somam 1. Use probabilidades "
            "intermediárias quando o texto for ambíguo, em vez de forçar certeza:\n%s"
            % (questions_block, state, schema)
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        error = None
        for _ in range(3):
            try:
                content = proxy.chat(task.teacher["provider"], messages, task.teacher.get("model"),
                                     temperature=task.teacher["temperature"])
                gold = parse_gold(task, content)
                return {"id": row["id"], "state": state, "state_tokens": n_tokens,
                        "questions": task.questions, "gold": gold,
                        "teacher": {"provider": task.teacher["provider"],
                                    "model": task.teacher.get("model")},
                        "source": row.get("source", "user")}, None
            except (ValueError, proxy.ProxyError) as exc:
                error = str(exc)
        return None, error

    with ThreadPoolExecutor(max_workers=task.teacher["concurrency"]) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        for i, future in enumerate(as_completed(futures), 1):
            result, error = future.result()
            if result:
                _append_jsonl(out_path, result, lock)
                stats["ok"] += 1
            else:
                _append_jsonl(fail_path, {"id": futures[future]["id"], "error": error}, lock)
                stats["failed"] += 1
            if i % 25 == 0 or i == len(todo):
                print("rotulados %d/%d (falhas: %d)" % (i, len(todo), stats["failed"]), flush=True)
    return stats


def generate_texts(task: Task, n: int, seed: int = 0,
                   words: Tuple[int, int] = (150, 1600)) -> int:
    """Gera n textos sintéticos em data/texts.jsonl. Para não gerar tudo na
    classe mais óbvia, cada pedido sorteia uma opção-alvo por pergunta; o
    professor depois rotula o texto de forma independente (a intenção do
    gerador não vira rótulo)."""
    rng = random.Random(seed)
    system = _read_prompt(task, "generator_prompt")
    out_path = task.path(task.data["texts"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lock = threading.Lock()

    def brief() -> Tuple[str, int]:
        hints = []
        for qid, q in task.questions.items():
            key = rng.choice(option_keys(q))
            if q["type"] == "choice":
                desc = q["criteria"][key]
                hints.append("- %s: %s%s" % (q["instructions"], key, " (%s)" % desc if desc else ""))
            elif q["type"] == "score":
                hints.append("- %s: nível %s (%s)" % (q["instructions"], key, q["criteria"][int(key)]))
            else:
                hints.append("- %s: %s" % (q["instructions"], "sim" if key == "true" else "não"))
        return "\n".join(hints), rng.randint(*words)

    briefs = [brief() for _ in range(n)]

    def work(item):
        hints, n_words = item
        user = (
            "Escreva UM texto novo, realista e em português do Brasil, com cerca de %d palavras, "
            "que corresponda a:\n%s\n\nVarie nomes, datas, estilo e estrutura. Não mencione "
            'estas instruções no texto. Responda APENAS com JSON: {"text": "..."}' % (n_words, hints)
        )
        content = proxy.chat(task.teacher["provider"],
                             [{"role": "system", "content": system}, {"role": "user", "content": user}],
                             task.teacher.get("model"), temperature=0.9,
                             max_tokens=min(8000, int(n_words * 2.5) + 200))
        match = re.search(r"\{.*\}", content, re.S)
        text = json.loads(match.group(0))["text"].strip() if match else ""
        return text, hints

    written = 0
    with ThreadPoolExecutor(max_workers=task.teacher["concurrency"]) as pool:
        for future in as_completed([pool.submit(work, b) for b in briefs]):
            try:
                text, hints = future.result()
            except (ValueError, KeyError, proxy.ProxyError) as error:
                print("falha ao gerar texto: %s" % error)
                continue
            if len(text) < 200:
                continue
            _append_jsonl(out_path, {"id": text_id(text), "state": text,
                                     "source": "synthetic", "hint": hints}, lock)
            written += 1
            if written % 25 == 0:
                print("gerados %d/%d" % (written, n), flush=True)
    return written


def iter_cases(path: str) -> Iterable[Dict[str, Any]]:
    yield from read_jsonl(path)

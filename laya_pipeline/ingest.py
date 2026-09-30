"""Converte o que a pessoa colocou no repositório da tarefa nos arquivos que
o pipeline usa. A pessoa não precisa escrever JSONL:

- data/documentos/   textos sem resposta: .txt/.md (um texto por arquivo),
                     .csv (coluna `texto`, opcional `id`) ou .jsonl (`state`
                     ou `texto` por linha). Subpastas são percorridas.
- data/exemplos/<opção>/   textos que a pessoa já respondeu: o nome da pasta
                     é a resposta (uma opção da pergunta do task.yaml).

Saída (fora do git, regenerável):
- data/texts.jsonl  todos os textos (documentos + exemplos + sintéticos já
                    gerados, que são preservados), para o professor;
- data/human.jsonl  as respostas humanas dos exemplos. Esses casos vão sempre
                    para a avaliação (nunca para o treino) e medem o professor
                    e o modelo contra uma pessoa.

Arquivos cujo nome começa com EXEMPLO- são ignorados: são as amostras que
vêm no template para mostrar o formato."""

import csv
import json
import os
from typing import Any, Dict, Iterator, List, Tuple

from .task import Task, option_keys
from .teacher import read_jsonl, text_id

TEXT_EXT = (".txt", ".md")
SAMPLE_PREFIX = "EXEMPLO-"
MIN_CHARS = 20


def _walk(root: str) -> Iterator[str]:
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if not name.startswith(".") and name != "README.md":
                yield os.path.join(base, name)


def _read(path: str) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return f.read().strip()


def read_documents(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Textos de um arquivo, no formato {state, id?, origin}. Devolve também
    os avisos (arquivo ignorado e por quê)."""
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    rows: List[Dict[str, Any]] = []
    if ext in TEXT_EXT:
        rows.append({"state": _read(path)})
    elif ext == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            cols = {(c or "").strip().lower(): c for c in reader.fieldnames or []}
            if "texto" not in cols:
                return [], ["%s: CSV sem coluna 'texto' (colunas: %s)" % (path, ", ".join(cols))]
            for line in reader:
                row = {"state": (line.get(cols["texto"]) or "").strip()}
                if "id" in cols and (line.get(cols["id"]) or "").strip():
                    row["id"] = line[cols["id"]].strip()
                rows.append(row)
    elif ext == ".jsonl":
        with open(path, encoding="utf-8-sig") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                data = json.loads(line)
                row = {"state": str(data.get("state") or data.get("texto") or "").strip()}
                if data.get("id"):
                    row["id"] = str(data["id"])
                rows.append(row)
    else:
        return [], ["%s: formato não suportado (use .txt, .md, .csv ou .jsonl)" % path]
    kept = [r for r in rows if len(r["state"]) >= MIN_CHARS]
    warnings = []
    if len(kept) < len(rows):
        warnings.append("%s: %d texto(s) vazio(s) ou curto(s) demais ignorado(s)" % (path, len(rows) - len(kept)))
    for r in kept:
        r["origin"] = path
    return kept, warnings


def example_question(task: Task) -> Tuple[str, List[str]]:
    """A pergunta que as pastas de data/exemplos/ respondem: só faz sentido
    com uma pergunta (o recomendado — um modelo, uma pergunta)."""
    if len(task.questions) != 1:
        raise SystemExit("data/exemplos/<opção>/ só funciona com uma pergunta no task.yaml "
                         "(este tem %d)" % len(task.questions))
    qid, q = next(iter(task.questions.items()))
    keys = option_keys(q)
    if q["type"] == "score":
        keys = keys + list(q["criteria"])  # aceita o índice ou o nome do nível
    elif q["type"] == "noul":
        keys = keys + ["sim", "nao"]
    return qid, keys


def _answer_key(task: Task, qid: str, folder: str) -> str:
    q = task.questions[qid]
    if q["type"] == "score" and folder in q["criteria"]:
        return str(q["criteria"].index(folder))
    if q["type"] == "noul":
        return {"sim": "true", "nao": "false"}.get(folder, folder)
    return folder


def ingest(task: Task) -> Dict[str, Any]:
    docs_dir = task.path(task.data["documents"])
    examples_dir = task.path(task.data["examples"])
    texts: Dict[str, Dict[str, Any]] = {}
    human: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    ignored_samples = 0

    def add(row: Dict[str, Any], source: str) -> str:
        row.setdefault("id", text_id(row["state"]))
        row["source"] = source
        if row["id"] in texts and texts[row["id"]]["state"] != row["state"]:
            warnings.append("id repetido com textos diferentes: %s (%s e %s)" % (
                row["id"], texts[row["id"]]["origin"], row["origin"]))
        texts[row["id"]] = row
        return row["id"]

    if os.path.isdir(docs_dir):
        for path in _walk(docs_dir):
            if os.path.basename(path).startswith(SAMPLE_PREFIX):
                ignored_samples += 1
                continue
            rows, warns = read_documents(path)
            warnings += warns
            for row in rows:
                row["origin"] = os.path.relpath(path, task.root)
                add(row, "user")

    if os.path.isdir(examples_dir):
        folders = sorted(d for d in os.listdir(examples_dir)
                         if os.path.isdir(os.path.join(examples_dir, d)) and not d.startswith("."))
        real = {d: [p for p in _walk(os.path.join(examples_dir, d))
                    if not os.path.basename(p).startswith(SAMPLE_PREFIX)] for d in folders}
        ignored_samples += sum(1 for d in folders for p in _walk(os.path.join(examples_dir, d))
                               if os.path.basename(p).startswith(SAMPLE_PREFIX))
        used = {d: files for d, files in real.items() if files}
        if used:
            qid, valid = example_question(task)
            unknown = [d for d in used if d not in valid]
            if unknown:
                raise SystemExit("pastas em %s que não são opções da pergunta '%s': %s\n"
                                 "opções válidas: %s" % (task.data["examples"], qid,
                                                         ", ".join(unknown), ", ".join(valid)))
            for folder, files in used.items():
                answer = _answer_key(task, qid, folder)
                for path in files:
                    rows, warns = read_documents(path)
                    warnings += warns
                    for row in rows:
                        row["origin"] = os.path.relpath(path, task.root)
                        case_id = add(row, "human")
                        human[case_id] = {"id": case_id, "human": {qid: answer}, "origin": row["origin"]}

    # Textos sintéticos (gen-texts) não vêm de arquivos: preserva os que já existem.
    synthetic = [r for r in read_jsonl(task.path(task.data["texts"])) if r.get("source") == "synthetic"]
    for row in synthetic:
        texts.setdefault(row["id"], row)

    os.makedirs(os.path.dirname(task.path(task.data["texts"])), exist_ok=True)
    for path, rows in ((task.path(task.data["texts"]), texts.values()),
                       (task.path(task.data["human"]), human.values())):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_answer: Dict[str, int] = {}
    for h in human.values():
        for answer in h["human"].values():
            per_answer[answer] = per_answer.get(answer, 0) + 1
    return {"texts": len(texts), "documents": sum(1 for t in texts.values() if t["source"] == "user"),
            "synthetic": len(synthetic), "human_examples": len(human), "human_per_answer": per_answer,
            "ignored_template_samples": ignored_samples, "warnings": warnings}


def load_human(task: Task) -> Dict[str, Dict[str, str]]:
    """{id: {qid: resposta}} das respostas humanas (vazio se não houver)."""
    path = task.path(task.data["human"])
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row["human"] for row in map(json.loads, filter(str.strip, f))}

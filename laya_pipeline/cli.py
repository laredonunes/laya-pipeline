"""CLI do pipeline. Rode de dentro do repositório da tarefa (ou use --task).

  laya-pipeline validate                 valida o task.yaml
  laya-pipeline status                   etapas feitas / próxima
  laya-pipeline ingest                   data/documentos + data/exemplos → data/*.jsonl
  laya-pipeline gen-texts --n 500        textos sintéticos → data/texts.jsonl
  laya-pipeline label                    professor rotula → runs/<v>/labeled.jsonl
  laya-pipeline split                    treino/avaliação estáveis por id
  laya-pipeline sample --n 20            runs/<v>/sample.md para revisão
  laya-pipeline approve data --by NOME   portão 1 (libera o treino)
  laya-pipeline eval                     baseline (modelo base, contexto da tarefa)
  laya-pipeline train                    fine-tuning → runs/<v>/checkpoint
  laya-pipeline eval --trained           avalia o checkpoint treinado
  laya-pipeline approve report --by NOME portão 2 (libera a publicação)
  laya-pipeline publish                  checkpoint + relatórios → S3, reports/ no repo
  laya-pipeline sync push|pull           runs/<v> entre estação e notebook (via S3)
"""

import argparse
import json
import os
import sys

from . import __version__
from .task import TaskError, load_task


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=".", help="raiz do repositório da tarefa")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate")
    sub.add_parser("status")
    sub.add_parser("ingest")
    p = sub.add_parser("gen-texts")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-words", type=int, default=150)
    p.add_argument("--max-words", type=int, default=1600)
    p = sub.add_parser("label")
    p.add_argument("--limit", type=int)
    sub.add_parser("split")
    p = sub.add_parser("sample")
    p.add_argument("--n", type=int, default=20)
    p = sub.add_parser("approve")
    p.add_argument("gate", choices=["data", "report"])
    p.add_argument("--by", required=True)
    p.add_argument("--note", default="")
    p = sub.add_parser("eval")
    p.add_argument("--trained", action="store_true", help="avalia runs/<v>/checkpoint")
    p.add_argument("--checkpoint", help="outro diretório de checkpoint")
    p.add_argument("--device")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("train")
    p.add_argument("--device")
    p.add_argument("--max-steps", type=int, help="teste de fumaça: para após N passos")
    p = sub.add_parser("publish")
    p.add_argument("--force", action="store_true", help="publica mesmo sem atingir as metas")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("sync")
    p.add_argument("direction", choices=["push", "pull"])

    args = parser.parse_args(argv)
    try:
        task = load_task(args.task)
    except TaskError as error:
        sys.exit(str(error))

    if args.cmd == "validate":
        print("ok: %s %s — %d pergunta(s), max_len=%d, professor=%s" % (
            task.name, task.version, len(task.questions), task.context["max_len"],
            task.teacher["provider"]))
    elif args.cmd == "status":
        from .state import status
        steps = status(task)
        nxt = next((s["step"] for s in steps if not s["done"]), None)
        for s in steps:
            print("[%s] %s" % ("x" if s["done"] else " ", s["step"]))
        print("próximo: %s" % (nxt or "nada — versão publicada"))
    elif args.cmd == "ingest":
        from .ingest import ingest
        _print(ingest(task))
    elif args.cmd == "gen-texts":
        from .teacher import generate_texts
        print("gerados:", generate_texts(task, args.n, args.seed, (args.min_words, args.max_words)))
    elif args.cmd == "label":
        from .teacher import label
        _print(label(task, args.limit))
    elif args.cmd == "split":
        from .dataset import split, stats
        from .teacher import read_jsonl
        n_train, n_eval = split(task)
        print("treino: %d | avaliação: %d" % (n_train, n_eval))
        _print(stats(task, read_jsonl(os.path.join(task.run_dir, "train.jsonl"))))
    elif args.cmd == "sample":
        from .dataset import sample
        print("revise:", sample(task, args.n))
    elif args.cmd == "approve":
        from .state import approve
        approve(task, args.gate, args.by, args.note)
        print("aprovado: %s por %s" % (args.gate, args.by))
    elif args.cmd == "eval":
        from .evaluate import evaluate
        checkpoint = args.checkpoint or (os.path.join(task.run_dir, "checkpoint") if args.trained else None)
        result = evaluate(task, "trained" if (args.trained or args.checkpoint) else "baseline",
                          checkpoint, args.device, args.limit)
        _print({"overall": result["overall"], "latency_s": result["latency_s"],
                "report": os.path.join(task.run_dir, "report.md")})
    elif args.cmd == "train":
        from .state import require
        from .train import train
        if args.max_steps is None:
            require(task, "data")
        train(task, args.device, args.max_steps)
    elif args.cmd == "publish":
        from .publish import publish
        from .state import require
        if not args.dry_run:
            require(task, "report")
        _print(publish(task, args.force, args.dry_run))
    elif args.cmd == "sync":
        from . import sync
        print("%s: %s" % (args.direction, ", ".join(getattr(sync, args.direction)(task)) or "nada"))


if __name__ == "__main__":
    main()

"""Leva os artefatos de uma rodada entre máquinas via S3 (pelo proxy).

A rotulagem roda onde o proxy responde rápido (estação); o treino roda onde
há GPU (Studio Lab/Kaggle/job). runs/ fica fora do git, então:
  estação:   laya-pipeline sync push     (dados rotulados + aprovações)
  notebook:  laya-pipeline sync pull  →  train / eval --trained / publish
  notebook:  laya-pipeline sync push     (relatórios de avaliação)
  estação:   laya-pipeline sync pull
O checkpoint não passa por aqui: vai direto para models/ no `publish`."""

import os
import urllib.parse
from typing import List

from . import proxy
from .task import Task

FILES = ("texts.jsonl", "labeled.jsonl", "train.jsonl", "eval.jsonl", "sample.md", "approvals.json",
         "eval_baseline.json", "eval_trained.json", "report.md", "train_log.jsonl")


def _prefix(task: Task) -> str:
    return "runs/laya/%s/%s/" % (task.name, task.version)


def push(task: Task) -> List[str]:
    sent = []
    for name in FILES:
        local = task.path(task.data["texts"]) if name == "texts.jsonl" else os.path.join(task.run_dir, name)
        if os.path.exists(local):
            proxy.s3_upload(local, _prefix(task) + name)
            sent.append(name)
    return sent


def pull(task: Task) -> List[str]:
    os.makedirs(task.run_dir, exist_ok=True)
    listing = proxy.request_json("GET", "/s3/?max=1000&prefix=" + urllib.parse.quote(_prefix(task)))
    keys = {obj["key"] for obj in listing["objects"]}
    got = []
    for name in FILES:
        key = _prefix(task) + name
        if key not in keys:
            continue
        local = task.path(task.data["texts"]) if name == "texts.jsonl" else os.path.join(task.run_dir, name)
        proxy.s3_download(key, local)
        got.append(name)
    return got

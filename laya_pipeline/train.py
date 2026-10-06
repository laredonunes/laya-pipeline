"""Fine-tuning RLCD do Laya numa GPU (ou CPU, só para teste de fumaça).

Adaptado de `research/scripts/finetune_single_device.py` do repositório
NandhaKishorM/laya (Apache-2.0; ver NOTICE). Mudanças em relação ao original:
- max_len / head_max_len / hiperparâmetros vêm do task.yaml (o original fixa 1024/256);
- acumulação de gradiente (micro-batch pequeno com contexto longo numa T4);
- checkpoint de estado a cada época e retomada automática (sessões do
  Studio Lab/Kaggle caem no meio do treino);
- pula itens que não cabem (opções estourando head_max_len) em vez de abortar.

Formato de entrada: runs/<versão>/train.jsonl, uma linha por caso
{"state", "questions", "gold"} — o mesmo esquema de docs/finetune.md do upstream."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from .hf import base_model_dir
from .task import Task
from .teacher import read_jsonl


def build_training_item(tok, cfg, state, q, gold_q):
    from laya.common import QTYPES, build_sequence, render_options

    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    else:
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(len(crit))]
    total = sum(target)
    target = [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)
    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                  cfg["max_len"], cfg["head_max_len"])
    if len(markers) != len(render_options({"t": t, "crit": crit})):
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target}


def preprocess(tok, cfg, rows: List[Dict[str, Any]]):
    items, skipped = [], 0
    for row in rows:
        for qid, q in row["questions"].items():
            if qid not in row["gold"]:
                continue
            item = build_training_item(tok, cfg, row["state"], q, row["gold"][qid])
            if item is None:
                skipped += 1
            else:
                item["qid"] = qid          # o peso de classe é por pergunta
                items.append(item)
    return items, skipped


def option_weights(items: List[Dict[str, Any]], mode: str = "none") -> List[List[float]]:
    """Peso por opção de cada item (função pura: dá para testar sem torch).

    `none` (padrão) devolve 1.0 para tudo — o comportamento de sempre.
    `auto` equilibra as OPÇÕES da pergunta pela massa que elas têm no conjunto:
    w_o = média(M)/M_o, onde M_o é a soma da probabilidade da opção o nos alvos
    do professor. Com o objetivo sendo soft-CE contra essas probabilidades, o
    modelo aprende o prior do TREINO; quando a régua tem outro prior (medido:
    29% de positivos no treino contra ~80% na avaliação), sem peso ele erra por
    baixo — e o erro caro desta tarefa é o falso negativo.
    """
    pesos = [[1.0] * len(it["target"]) for it in items]
    if mode == "none":
        return pesos
    massa: Dict[str, List[float]] = {}
    for it in items:
        acc = massa.setdefault(it["qid"], [0.0] * len(it["target"]))
        for i, valor in enumerate(it["target"]):
            acc[i] += valor
    fator = {qid: [(sum(acc) / len(acc)) / v if v > 0 else 1.0 for v in acc]
             for qid, acc in massa.items()}
    for it, linha in zip(items, pesos):
        f = fator[it["qid"]]
        for i in range(len(linha)):
            linha[i] = f[i] if i < len(f) else 1.0
    return pesos


def weights_by_question(items: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Um exemplo de vetor de peso por pergunta (para imprimir no log)."""
    vistos: Dict[str, List[float]] = {}
    for it in items:
        if it["qid"] not in vistos and it.get("weight"):
            vistos[it["qid"]] = [round(float(x), 3) for x in it["weight"]]
    return vistos


def collate(items, pad_id):
    import torch

    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    weight = torch.ones((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
        if it.get("weight"):
            weight[i, : len(it["weight"])] = torch.tensor(it["weight"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "weight": weight, "qtype": torch.tensor([it["qtype"] for it in items])}


def _forward(model, batch, device, use_amp):
    import torch

    args = [batch[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos",
                                          "marker_mask", "qtype")]
    if use_amp:
        with torch.autocast("cuda", dtype=torch.float16):
            return model(*args)
    return model(*args)


def fit_one_temp(sel):
    import torch

    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError:
        return 1.2
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def train(task: Task, device: Optional[str] = None, max_steps: Optional[int] = None) -> str:
    """Treina e grava runs/<versão>/checkpoint/ (carregável por laya.load).
    `max_steps` limita o treino (teste de fumaça do pipeline em CPU)."""
    import torch
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    hp = task.train
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_amp = device.type == "cuda"
    torch.manual_seed(hp["seed"])
    random.seed(hp["seed"])
    print("dispositivo:", device, flush=True)

    model_dir = base_model_dir(task.base_model["id"], task.base_model.get("subfolder"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["max_len"] = task.context["max_len"]
    cfg["head_max_len"] = task.context["head_max_len"]

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if use_amp:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()

    rows = read_jsonl(os.path.join(task.run_dir, "train.jsonl"))
    if not rows:
        raise SystemExit("train.jsonl vazio: rode `split` primeiro")
    all_items, skipped = preprocess(tok, cfg, rows)
    if skipped:
        print("aviso: %d itens pulados (opções não cabem em head_max_len=%d)"
              % (skipped, cfg["head_max_len"]), flush=True)

    # Fatia de calibração separada ANTES do treino (ver docs/finetune.md do
    # upstream): ajustar a temperatura em itens já vistos degenera a escala.
    order = list(range(len(all_items)))
    random.Random(hp["seed"]).shuffle(order)
    n_calib = min(400, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]

    micro, accum, group_size, epochs = hp["micro_batch"], hp["grad_accum"], 4, hp["epochs"]
    peso_modo = str(hp.get("class_weight", "none"))
    for item, peso in zip(train_items, option_weights(train_items, peso_modo)):
        item["weight"] = peso
    if peso_modo != "none":
        for qid, vetor in weights_by_question(train_items).items():
            print("peso de classe (%s) %s: %s" % (peso_modo, qid, vetor), flush=True)
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": hp["lr_encoder"]},
                                   {"params": head_params, "lr": hp["lr_head"]}], weight_decay=0.01)
    micro_per_epoch = max(1, (len(train_items) + micro - 1) // micro)
    steps_per_epoch = max(1, (micro_per_epoch + accum - 1) // accum)
    total_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    state_path = os.path.join(task.run_dir, "train_state.pt")
    start_epoch = 0
    hyperparams = {k: hp[k] for k in sorted(hp)}
    if os.path.exists(state_path) and max_steps is None:
        state = torch.load(state_path, map_location=device, weights_only=False)
        anterior = state.get("hyperparams")
        if anterior is not None and anterior != hyperparams:
            # O estado carrega pesos treinados com outro objetivo (epochs, peso de
            # classe, lr...). Retomar dali treinaria o novo com a mistura do velho:
            # começa do zero e diz por quê.
            print("train_state.pt é de outra configuração (%s)\n  agora: %s\n  começando do zero"
                  % (anterior, hyperparams), flush=True)
            state = None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        print("retomando da época %d" % (start_epoch + 1), flush=True)

    print("itens de treino: %d (%d para calibração) | %d épocas | %d passos/época "
          "(micro-batch %d × acumulação %d)" % (len(train_items), len(calib_items), epochs,
                                                steps_per_epoch, micro, accum), flush=True)
    log_path = os.path.join(task.run_dir, "train_log.jsonl")
    steps_done = start_epoch * steps_per_epoch
    if start_epoch == 0 and max_steps is None:
        # O log é de UMA rodada: o arquivo volta pelo `sync pull` com as épocas da
        # rodada anterior e, sem isto, o resumo mistura as duas (na 0.3.0 ficaram
        # 48 linhas para 4 épocas de uma rodada e 24 da seguinte).
        open(log_path, "w").close()
    for epoch in range(start_epoch, epochs):
        random.Random(hp["seed"] + epoch).shuffle(train_items)
        sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, epochs - 1))
        epoch_loss, n_micro, started = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for m_idx, b_idx in enumerate(range(0, len(train_items), micro)):
            batch = collate(train_items[b_idx: b_idx + micro], tok.pad_token_id)
            logits, _ = _forward(model, batch, device, use_amp)
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            # Soft-CE ponderada pelo peso de classe da opção. Com peso 1.0 (o
            # padrão) o resultado é idêntico ao de antes: a divisão por
            # (target*peso).sum() só existe para a escala da perda não mudar
            # quando o equilíbrio está ligado.
            peso = batch["weight"].to(device)
            alvo = (target * peso) / (target * peso).sum(-1, keepdim=True).clamp_min(1e-6)
            loss = (-(adv * logp).mean()
                    - (alvo * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean())

            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            epoch_loss += loss.item()
            n_micro += 1
            last_micro = b_idx + micro >= len(train_items)
            if (m_idx + 1) % accum == 0 or last_micro:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps_done += 1
                if steps_done % 20 == 0:
                    print("época %d passo %d/%d | loss %.4f | %.0fs" % (
                        epoch + 1, steps_done - epoch * steps_per_epoch, steps_per_epoch,
                        epoch_loss / n_micro, time.time() - started), flush=True)
                if max_steps is not None and steps_done >= max_steps:
                    break
        avg = epoch_loss / max(1, n_micro)
        print("época %d/%d | loss média %.4f | %.0fs" % (epoch + 1, epochs, avg, time.time() - started),
              flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "loss": round(avg, 5),
                                "seconds": round(time.time() - started)}) + "\n")
        if max_steps is not None and steps_done >= max_steps:
            break
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None, "epoch": epoch,
                    "hyperparams": hyperparams}, state_path)

    model.eval()
    calib_preds = []
    with torch.no_grad():
        for c_idx in range(0, len(calib_items), 8):
            chunk = calib_items[c_idx: c_idx + 8]
            logits, _ = _forward(model, collate(chunk, tok.pad_token_id), device, use_amp)
            l_np = logits.float().cpu().numpy()
            for r, it in enumerate(chunk):
                calib_preds.append((it["qtype"], l_np[r, : len(it["markers"])], it["target"]))
    fitted = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
        if sel:
            fitted[qt] = fit_one_temp(sel)
    print("temperaturas ajustadas (choice, score, noul):", [round(t, 3) for t in fitted], flush=True)

    out_dir = os.path.join(task.run_dir, "checkpoint")
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    cfg.pop("temperature_by_options", None)  # mascararia a calibração nova
    cfg.update({"fine_tuned": True, "temperature": fitted, "task": task.name,
                "task_version": task.version,
                "training": {"epochs": epochs, "train_items": len(train_items),
                             "calib_items": len(calib_items), "smoke_test": max_steps is not None,
                             "class_weight": peso_modo}})
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if os.path.exists(state_path) and max_steps is None:
        os.remove(state_path)  # treino concluído; o estado (~3–4 GB) não é mais necessário
    print("checkpoint salvo em", out_dir, flush=True)
    return out_dir

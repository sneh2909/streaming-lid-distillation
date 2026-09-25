"""Distil a frozen teacher into the streaming student.

Loss at every target frame t (a teacher query point) of every clip:
    KL( q_t || softmax(z_t) ) = sum_k q_tk (log q_tk - log p_tk)
averaged over valid target frames. q_t comes from the chosen target kind (see slid/targets.py);
the chunk size is resampled every batch so one model serves several lookaheads.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slid.audio import SR, load_wav
from slid.config import LANGS
from slid.student import StreamingLID, frame_end_sample

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = (1, 2, 4, 8)                     # 80 / 160 / 320 / 640 ms


def read_jsonl(name):
    return [json.loads(l) for l in open(ROOT / f"data/manifests/{name}.jsonl", encoding="utf-8")]


def make_batch(clips, targets, max_s):
    """Clips are only truncated at the END: cutting the start would change what the
    causal student has heard relative to what the prefix/causal teacher heard."""
    xs = [c[: int(max_s * SR)] for c in clips["wav"]]
    n = max(len(x) for x in xs)
    wav = torch.zeros(len(xs), n)
    for i, x in enumerate(xs):
        wav[i, : len(x)] = torch.from_numpy(x)
    rows, frames, qs = [], [], []
    for i, p in enumerate(clips["path"]):
        f, q = targets[p]
        keep = torch.tensor([frame_end_sample(int(t)) <= len(xs[i]) for t in f], dtype=torch.bool)
        rows.append(torch.full((int(keep.sum()),), i))
        frames.append(f[keep])
        qs.append(q[keep])
    return wav, torch.cat(rows), torch.cat(frames), torch.cat(qs)


def kd_loss(logits, rows, frames, q):
    logp = F.log_softmax(logits[rows, frames].float(), dim=-1)
    return (q * (torch.log(q.clamp_min(1e-8)) - logp)).sum(-1).mean()


@torch.no_grad()
def fit_cmvn(model, wavs, device):
    feats = torch.cat([model.feat.mel(torch.from_numpy(w).to(device)[None]).add(1e-6).log()[0].T for w in wavs])
    model.feat.mean.copy_(feats.mean(0))
    model.feat.std.copy_(feats.std(0).clamp_min(1e-3))


@torch.no_grad()
def evaluate(model, data, targets, device, chunk=4, max_s=20.0):
    model.eval()
    losses, agree, n = [], 0, 0
    for i in range(0, len(data["path"]), 16):
        clips = {k: v[i: i + 16] for k, v in data.items()}
        wav, rows, frames, q = make_batch(clips, targets, max_s)
        logits = model(wav.to(device), chunk)
        rows, frames, q = rows.to(device), frames.to(device), q.to(device)
        losses.append(kd_loss(logits, rows, frames, q).item() * len(rows))
        agree += (logits[rows, frames].argmax(-1) == q.argmax(-1)).sum().item()
        n += len(rows)
    model.train()
    return sum(losses) / n, agree / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--kind", default="causal")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-s", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    targets = torch.load(ROOT / f"data/targets/{args.teacher}/{args.kind}.pt")
    train = read_jsonl("train") + read_jsonl("train_mix")
    held = read_jsonl("eval") + read_jsonl("switch")
    load = lambda items: {"path": [it["path"] for it in items], "wav": [load_wav(it["path"]) for it in items]}
    train_d, held_d = load(train), load(held)

    model = StreamingLID(len(LANGS)).to(args.device)
    fit_cmvn(model, random.sample(train_d["wav"], 200), args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    warm = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    log = {"args": vars(args), "n_params": n_params, "n_train": len(train), "n_held": len(held),
           "steps": [], "eval": []}
    print(f"params {n_params/1e6:.2f}M  train {len(train)} clips  held-out {len(held)} clips")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = random.sample(range(len(train)), args.batch)
        clips = {k: [v[i] for i in idx] for k, v in train_d.items()}
        chunk = random.choice(CHUNKS)
        wav, rows, frames, q = make_batch(clips, targets, args.max_s)
        logits = model(wav.to(args.device), chunk)
        loss = kd_loss(logits, rows.to(args.device), frames.to(args.device), q.to(args.device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
        opt.step(); sched.step()
        log["steps"].append({"step": step, "loss": loss.item(), "grad_norm": gnorm, "chunk": chunk})
        if step % 50 == 0:
            print(f"step {step} loss {np.mean([s['loss'] for s in log['steps'][-50:]]):.4f} "
                  f"gnorm {gnorm:.2f} {time.time() - t0:.0f}s")
        if step % args.eval_every == 0 or step == args.steps:
            hl, ha = evaluate(model, held_d, targets, args.device)
            log["eval"].append({"step": step, "heldout_kd": hl, "heldout_agree": ha})
            print(f"  held-out KD {hl:.4f}  agreement {ha:.3f}")

    out = Path(args.out or ROOT / f"checkpoints/{args.teacher}_{args.kind}")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "n_langs": len(LANGS), "args": vars(args)}, out / "student.pt")
    (out / "train_log.json").write_text(json.dumps(log))
    print("saved", out)


if __name__ == "__main__":
    main()

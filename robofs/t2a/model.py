# Copyright (c) 2026, the RoboFS authors. RoboFS Non-Commercial Research License (see LICENSE).
# Portions copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md), BSD-3-Clause.
# All rights reserved.

"""RoboFS-T2A: a text-to-action Transformer trained on RoboFS-Data shards (``robofs.t2a.pack``).

A character Transformer encoder and an autoregressive decoder emit chunks of normalized robot commands (4 frames
per step) and a stop probability per chunk, so the sequence length and the timing of each letter are predicted.
Predictions are written as ``actions``, ``lengths`` and ``words`` and executed open loop in simulation with
``robofs.evaluation.evaluate --method replay``.

Usage:
    python -m robofs.t2a.model train --shards data/robofs_data/part_*.npz --output_dir checkpoints/t2a
    python -m robofs.t2a.model predict --checkpoint checkpoints/t2a/model.pt --words_file words.txt --output_file pred.npz
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robofs.config import path, section

ACTION_DIM = 18
ACTION_LAYOUT = "14 normalized joint positions (soft limits) + 4 normalized fixed-tendon positions"


def _positions(length: int, dim: int, device) -> torch.Tensor:
    """Sinusoidal position encoding [length, dim]."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding


class Text2ActionNet(nn.Module):
    """Characters [B, L] -> command chunks [B, C, chunk, 18] in [-1, 1] and stop logits [B, C]."""

    def __init__(self, vocab_size: int, chunk: int = 4, dim: int = 256, layers: int = 4, heads: int = 8):
        super().__init__()
        self.chunk, self.dim, self.pad = chunk, dim, vocab_size
        self.char_embedding = nn.Embedding(vocab_size + 1, dim, padding_idx=vocab_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.1, batch_first=True, norm_first=True), layers
        )
        self.start = nn.Parameter(torch.zeros(dim))
        self.chunk_embedding = nn.Linear(chunk * ACTION_DIM, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, 4 * dim, 0.1, batch_first=True, norm_first=True), layers
        )
        self.action_head = nn.Linear(dim, chunk * ACTION_DIM)
        self.stop_head = nn.Linear(dim, 1)

    def encode(self, chars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = chars == self.pad
        x = self.char_embedding(chars) + _positions(chars.shape[1], self.dim, chars.device)
        return self.encoder(x, src_key_padding_mask=padding), padding

    def decode(self, memory, memory_padding, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict chunk i from the chunks before i; ``previous`` [B, C, chunk, 18] holds the chunks to shift in."""
        batch, chunks = previous.shape[:2]
        inputs = self.chunk_embedding(previous[:, :-1].reshape(batch, chunks - 1, self.chunk * ACTION_DIM))
        x = torch.cat([self.start.expand(batch, 1, -1), inputs], dim=1) + _positions(chunks, self.dim, previous.device)
        causal = nn.Transformer.generate_square_subsequent_mask(chunks, device=previous.device)
        h = self.decoder(x, memory, tgt_mask=causal, memory_key_padding_mask=memory_padding, tgt_is_causal=True)
        actions = torch.tanh(self.action_head(h)).reshape(batch, chunks, self.chunk, ACTION_DIM)
        return actions, self.stop_head(h)[..., 0]


def encode_words(words: list[str], vocab: dict[str, int]) -> torch.Tensor:
    """Character ids [N, max_len], padded with ``len(vocab)``."""
    unknown = sorted({c for w in words for c in w} - set(vocab))
    if unknown:
        raise ValueError(f"Characters missing from the model vocabulary: {unknown}")
    chars = torch.full((len(words), max(map(len, words))), len(vocab), dtype=torch.long)
    for row, word in enumerate(words):
        chars[row, : len(word)] = torch.tensor([vocab[c] for c in word])
    return chars


def load_shards(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Concatenate shards into words, actions [N, T, 18] (last frame held over padding) and lengths [N]."""
    words, actions, lengths = [], [], []
    for shard in paths:
        with np.load(shard, allow_pickle=False) as data:
            words += data["words"].tolist()
            actions += [a[:n] for a, n in zip(data["actions"], data["lengths"])]
            lengths += data["lengths"].tolist()
    frames = max(lengths)
    padded = np.stack([np.concatenate([a, np.repeat(a[-1:], frames - len(a), axis=0)]) for a in actions])
    return words, padded.astype(np.float32), np.array(lengths)


def chunk_targets(actions: torch.Tensor, lengths: torch.Tensor, chunk: int):
    """Chunked targets [B, C, chunk, 18], frame validity [B, C, chunk], chunk validity [B, C] and stop labels [B, C]."""
    chunks = math.ceil(actions.shape[1] / chunk)
    frames = chunks * chunk
    actions = torch.cat([actions, actions[:, -1:].expand(-1, frames - actions.shape[1], -1)], dim=1)
    frame_valid = torch.arange(frames, device=actions.device)[None] < lengths[:, None]
    count = (lengths + chunk - 1) // chunk
    chunk_valid = torch.arange(chunks, device=actions.device)[None] < count[:, None]
    stop = torch.arange(chunks, device=actions.device)[None] == count[:, None] - 1
    shape = (len(actions), chunks, chunk)
    return actions.reshape(*shape, ACTION_DIM), frame_valid.reshape(shape), chunk_valid, stop.float()


def _loss(model, chars, actions, lengths, stop_weight: float, input_noise: float = 0.0):
    target, frame_valid, chunk_valid, stop = chunk_targets(actions, lengths, model.chunk)
    memory, padding = model.encode(chars)
    # Noisy teacher-forced inputs reduce error accumulation when decoding from the model's own chunks.
    previous = target + input_noise * torch.randn_like(target) if input_noise > 0 else target
    prediction, stop_logits = model.decode(memory, padding, previous)
    mse = ((prediction - target).square().mean(-1) * frame_valid).sum() / frame_valid.sum()
    bce = nn.functional.binary_cross_entropy_with_logits(stop_logits, stop, reduction="none")
    bce = (bce * chunk_valid).sum() / chunk_valid.sum()
    return mse + stop_weight * bce, mse, bce


@torch.no_grad()
def generate(model, chars: torch.Tensor, max_chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy decoding until each sequence's stop probability exceeds 0.5 or it reaches ``max_chunks``."""
    memory, padding = model.encode(chars)
    batch = len(chars)
    chunks = torch.zeros(batch, 1, model.chunk, ACTION_DIM, device=chars.device)
    count = torch.zeros(batch, dtype=torch.long, device=chars.device)
    done = torch.zeros(batch, dtype=torch.bool, device=chars.device)
    for step in range(int(max_chunks.max())):
        # The last slot is a placeholder: decode() predicts every slot from the chunks before it.
        prediction, stop_logits = model.decode(memory, padding, chunks)
        chunks[:, step] = prediction[:, step]
        count = torch.where(done, count, count + 1)
        done |= (torch.sigmoid(stop_logits[:, step]) > 0.5) | (count >= max_chunks)
        if done.all():
            break
        chunks = torch.cat([chunks, torch.zeros_like(chunks[:, :1])], dim=1)
    actions = chunks.reshape(batch, -1, ACTION_DIM)
    return actions, count * model.chunk


def train(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    words, actions, lengths = load_shards(args.shards)
    order = np.random.default_rng(args.seed).permutation(len(words))
    val_index, train_index = order[: args.val_words], order[args.val_words :]
    vocab = {c: i for i, c in enumerate(sorted({c for w in words for c in w}))}
    chars = encode_words(words, vocab)
    actions_t, lengths_t = torch.as_tensor(actions), torch.as_tensor(lengths)
    config = {"vocab": vocab, "chunk": args.chunk, "dim": args.dim, "layers": args.layers, "heads": args.heads}
    model = Text2ActionNet(len(vocab), args.chunk, args.dim, args.layers, args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps, pct_start=0.05)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, start = [], time.time()
    for step in range(1, args.steps + 1):
        index = torch.as_tensor(train_index)[torch.randint(0, len(train_index), (args.batch_size,), generator=generator)]
        n = lengths_t[index].to(device)
        frames = int(n.max())
        x, y = chars[index].to(device), actions_t[index, :frames].to(device)
        loss, _, _ = _loss(model, x, y, n, args.stop_weight, args.input_noise)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.5f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, chars, actions_t, lengths_t, val_index, args.stop_weight, device)
            history.append({"step": step, "train_loss": loss.item(), **val, "minutes": (time.time() - start) / 60})
            print(json.dumps(history[-1]), flush=True)
    summary = {"train_words": len(train_index), "val_words": len(val_index), "shards": [str(p) for p in args.shards]}
    torch.save({"model": model.state_dict(), "config": config, **summary}, args.output_dir / "model.pt")
    (args.output_dir / "history.json").write_text(json.dumps({**summary, "history": history}, indent=2) + "\n")


@torch.no_grad()
def evaluate(model, chars, actions, lengths, index, stop_weight, device, batch_size: int = 256) -> dict:
    """Teacher-forced MSE and stop loss, plus free-running MSE and length accuracy on held-out words."""
    model.eval()
    sums = np.zeros(4)
    for start in range(0, len(index), batch_size):
        rows = torch.as_tensor(index[start : start + batch_size])
        n = lengths[rows].to(device)
        x, y = chars[rows].to(device), actions[rows, : int(n.max())].to(device)
        _, mse, bce = _loss(model, x, y, n, stop_weight)
        predicted, predicted_lengths = generate(model, x, max_chunks=(2 * n + model.chunk - 1) // model.chunk)
        frames = min(predicted.shape[1], y.shape[1])
        valid = torch.arange(frames, device=device)[None] < torch.minimum(n, predicted_lengths)[:, None]
        free_mse = ((predicted[:, :frames] - y[:, :frames]).square().mean(-1) * valid).sum() / valid.sum()
        sums += np.array([mse.item(), bce.item(), free_mse.item(), (predicted_lengths == n).float().mean().item()])
    model.train()
    batches = math.ceil(len(index) / batch_size)
    keys = ("val_mse", "val_stop_bce", "val_free_running_mse", "val_length_accuracy")
    return dict(zip(keys, (sums / batches).tolist()))


def load_model(checkpoint: Path, device: torch.device) -> tuple[Text2ActionNet, dict[str, int]]:
    """Load a trained model in evaluation mode and its character vocabulary."""
    state = torch.load(checkpoint, map_location=device)
    config = state["config"]
    vocab = config["vocab"]
    model = Text2ActionNet(len(vocab), config["chunk"], config["dim"], config["layers"], config["heads"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, vocab


@torch.no_grad()
def predict(args) -> None:
    device = torch.device(args.device)
    model, vocab = load_model(args.checkpoint, device)
    lines = args.words_file.read_text(encoding="utf-8").splitlines()
    words = list(dict.fromkeys(w.strip().lower() for w in lines if w.strip()))
    chars = encode_words(words, vocab)
    outputs, lengths = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for i in range(0, len(words), args.batch_size):
        x = chars[i : i + args.batch_size].to(device)
        letters = (x != len(vocab)).sum(1)
        max_chunks = (args.max_frames_per_letter * letters + model.chunk - 1) // model.chunk
        actions, n = generate(model, x, max_chunks)
        outputs += [a[:k].cpu().numpy() for a, k in zip(actions, n)]
        lengths += n.tolist()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    frames = max(lengths)
    actions = np.stack([np.concatenate([a, np.repeat(a[-1:], frames - len(a), axis=0)]) for a in outputs])
    metadata = {
        "words_file": str(args.words_file.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "action_layout": ACTION_LAYOUT,
        "fps": 30.0,
        "inference_seconds": seconds,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_file,
        actions=actions.astype(np.float32),
        lengths=np.array(lengths),
        words=np.array(words),
        metadata=np.array(json.dumps(metadata)),
    )
    print(json.dumps({"words": len(words), "inference_seconds": seconds}), flush=True)


def main() -> None:
    cfg = section("t2a")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train on RoboFS-Data shards.")
    t.add_argument("--shards", type=Path, nargs="+", required=True)
    t.add_argument("--output_dir", type=Path, required=True)
    t.add_argument("--val_words", type=int, default=cfg["val_words"], help="Held-out shard words for validation losses.")
    t.add_argument("--chunk", type=int, default=cfg["chunk"], help="Frames predicted per decoder step.")
    t.add_argument("--dim", type=int, default=cfg["dim"])
    t.add_argument("--layers", type=int, default=cfg["layers"])
    t.add_argument("--heads", type=int, default=cfg["heads"])
    t.add_argument("--steps", type=int, default=cfg["steps"])
    t.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    t.add_argument("--lr", type=float, default=cfg["learning_rate"])
    t.add_argument("--stop_weight", type=float, default=cfg["stop_weight"])
    t.add_argument("--input_noise", type=float, default=cfg["input_noise"], help="Std of noise on teacher-forced chunks.")
    t.add_argument("--eval_every", type=int, default=cfg["eval_every"])
    t.add_argument("--seed", type=int, default=cfg["seed"])
    t.add_argument("--device", default=device)
    p = sub.add_parser("predict", help="Predict commands for a list of words.")
    p.add_argument("--checkpoint", type=Path, default=path("t2a_checkpoint"))
    p.add_argument("--words_file", type=Path, required=True)
    p.add_argument("--output_file", type=Path, required=True)
    p.add_argument(
        "--max_frames_per_letter",
        type=int,
        default=cfg["max_frames_per_letter"],
        help="Decoding limit if no stop is predicted.",
    )
    p.add_argument("--batch_size", type=int, default=cfg["predict_batch_size"])
    p.add_argument("--device", default=device)
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"SSRTRC01"
HEADER = struct.Struct("<8s6H3I")
HIDDEN_MAGIC = b"SSHID001"
HIDDEN_HEADER = struct.Struct("<8sII")


@dataclass(frozen=True)
class Header:
    version: int
    header_bytes: int
    record_bytes: int
    n_layers: int
    top_k: int
    flags: int
    prompt_tokens: int
    decode_tokens: int


def read_trace(path: Path) -> tuple[Header, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"short trace header: {path}")
    fields = HEADER.unpack_from(raw)
    if fields[0] != MAGIC:
        raise ValueError(f"bad trace magic: {fields[0]!r}")
    header = Header(*fields[1:-1])
    expected_record_bytes = 12 + 6 * header.top_k
    if header.header_bytes != HEADER.size or header.record_bytes != expected_record_bytes:
        raise ValueError("trace layout does not match header")
    payload = memoryview(raw)[header.header_bytes:]
    if len(payload) % header.record_bytes:
        raise ValueError("trace payload has a partial record")
    dtype = np.dtype(
        [
            ("token_index", "<u4"),
            ("token_id", "<i4"),
            ("layer_id", "<u2"),
            ("phase", "u1"),
            ("top_k", "u1"),
            ("expert_ids", "<i2", (header.top_k,)),
            ("weights", "<f4", (header.top_k,)),
        ],
        align=False,
    )
    records = np.frombuffer(payload, dtype=dtype)
    if dtype.itemsize != header.record_bytes:
        raise ValueError("numpy record layout mismatch")
    if len(records) and not np.all(records["top_k"] == header.top_k):
        raise ValueError("record top_k differs from header")
    return header, records


def write_fixture(path: Path, tokens: int = 4, layers: int = 3, top_k: int = 10) -> None:
    record_bytes = 12 + 6 * top_k
    with path.open("wb") as out:
        out.write(HEADER.pack(MAGIC, 1, HEADER.size, record_bytes, layers, top_k, 0, 0, tokens, 0))
        for token in range(tokens):
            for layer in range(layers):
                ids = [(token * 17 + layer * 11 + i) % 512 for i in range(top_k)]
                weights = np.arange(top_k, 0, -1, dtype=np.float32)
                weights /= weights.sum()
                out.write(struct.pack("<IiHBB", token, 1000 + token, layer, 1, top_k))
                out.write(np.asarray(ids, dtype="<i2").tobytes())
                out.write(weights.astype("<f4").tobytes())


def read_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_hidden(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < HIDDEN_HEADER.size:
        raise ValueError(f"short hidden-state header: {path}")
    magic, n_embd, rows = HIDDEN_HEADER.unpack_from(raw)
    if magic != HIDDEN_MAGIC or n_embd == 0:
        raise ValueError("hidden-state layout does not match header")
    record_bytes = 4 + n_embd * 4
    payload = memoryview(raw)[HIDDEN_HEADER.size:]
    if len(payload) != rows * record_bytes:
        raise ValueError("hidden-state payload size does not match header")
    token_indices = np.empty(rows, dtype=np.uint32)
    hidden = np.empty((rows, n_embd), dtype=np.float32)
    for row in range(rows):
        offset = row * record_bytes
        token_indices[row] = struct.unpack_from("<I", payload, offset)[0]
        hidden[row] = np.frombuffer(payload, dtype="<f4", count=n_embd, offset=offset + 4)
    return token_indices, hidden

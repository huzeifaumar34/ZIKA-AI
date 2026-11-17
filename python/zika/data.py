"""Validacao e preparacao de corpus JSONL para o ZIKA-AI.

Formato de cada linha:
{"turns":[{"role":"user","text":"..."},{"role":"ia","text":"..."}]}

Os binarios guardam IDs uint32 little-endian, sem padding. A divisao e feita
por conversa, nunca por token, para evitar fuga entre treino e validacao.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from .tokenizer import CharTokenizer, SPECIAL_TOKENS, TokenizerError

ROLE_TOKEN = {"user": "<|user|>", "ia": "<|ia|>"}


class DatasetError(ValueError):
    """Corpus JSONL invalido ou nao representavel."""


def validate_conversation(
    item: Any, *, source: str = "<memory>", line_number: int = 1,
    tokenizer: CharTokenizer | None = None,
) -> dict[str, list[dict[str, str]]]:
    tokenizer = tokenizer or CharTokenizer()
    if not isinstance(item, dict) or set(item) != {"turns"}:
        raise DatasetError(f"{source}:{line_number}: cada registo deve conter apenas a chave 'turns'")
    turns = item["turns"]
    if not isinstance(turns, list) or len(turns) < 2:
        raise DatasetError(f"{source}:{line_number}: 'turns' tem de conter pelo menos duas mensagens")

    clean_turns: list[dict[str, str]] = []
    expected_role = "user"
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or set(turn) != {"role", "text"}:
            raise DatasetError(
                f"{source}:{line_number}: turno {index + 1} deve conter 'role' e 'text'"
            )
        role, text = turn["role"], turn["text"]
        if not isinstance(role, str) or role not in ROLE_TOKEN:
            raise DatasetError(f"{source}:{line_number}: role invalido no turno {index + 1}: {role!r}")
        if role != expected_role:
            raise DatasetError(
                f"{source}:{line_number}: esperado role {expected_role!r} no turno {index + 1}"
            )
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(f"{source}:{line_number}: texto vazio ou invalido no turno {index + 1}")
        if any(marker in text for marker in SPECIAL_TOKENS):
            raise DatasetError(
                f"{source}:{line_number}: texto nao pode conter marcadores especiais reservados"
            )
        try:
            tokenizer.encode(text, parse_special_tokens=False)
        except TokenizerError as exc:
            raise DatasetError(f"{source}:{line_number}: {exc}") from exc
        clean_turns.append({"role": role, "text": text})
        expected_role = "ia" if expected_role == "user" else "user"

    if clean_turns[-1]["role"] != "ia":
        raise DatasetError(f"{source}:{line_number}: a conversa tem de terminar com uma resposta da ia")
    return {"turns": clean_turns}


def load_jsonl(
    path: str | Path, tokenizer: CharTokenizer | None = None
) -> list[dict[str, list[dict[str, str]]]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"nao foi possivel ler {source}: {exc}") from exc

    conversations: list[dict[str, list[dict[str, str]]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"{source}:{line_number}: JSON invalido na coluna {exc.colno}: {exc.msg}"
            ) from exc
        conversations.append(
            validate_conversation(
                item, source=str(source), line_number=line_number, tokenizer=tokenizer
            )
        )
    return conversations


def serialize_conversation(conversation: dict[str, list[dict[str, str]]]) -> str:
    pieces: list[str] = []
    for turn in conversation["turns"]:
        pieces.extend((ROLE_TOKEN[turn["role"]], turn["text"]))
    pieces.append("<|fim|>")
    return "".join(pieces)


def encode_conversation(
    conversation: dict[str, list[dict[str, str]]], tokenizer: CharTokenizer | None = None
) -> list[int]:
    tokenizer = tokenizer or CharTokenizer()
    return tokenizer.encode(serialize_conversation(conversation))


def _encode_many(
    conversations: Sequence[dict[str, list[dict[str, str]]]], tokenizer: CharTokenizer
) -> list[int]:
    result: list[int] = []
    for conversation in conversations:
        result.extend(encode_conversation(conversation, tokenizer))
    return result


def _write_u32le(path: Path, values: Iterable[int]) -> tuple[int, str]:
    payload = bytearray()
    count = 0
    for value in values:
        if value < 0 or value > 0xFFFFFFFF:
            raise DatasetError(f"ID de token fora de uint32: {value}")
        payload.extend(struct.pack("<I", value))
        count += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return count, hashlib.sha256(payload).hexdigest()


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0.0 < validation_ratio < 1.0:
        raise DatasetError("validation_ratio tem de estar entre 0 e 1")
    tokenizer = CharTokenizer()
    conversations = load_jsonl(input_path, tokenizer)
    if len(conversations) < 2:
        raise DatasetError("sao necessarias pelo menos duas conversas para dividir treino e validacao")

    indices = list(range(len(conversations)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, int(round(len(indices) * validation_ratio)))
    validation_count = min(validation_count, len(indices) - 1)
    validation_indices = set(indices[:validation_count])

    train = [item for index, item in enumerate(conversations) if index not in validation_indices]
    validation = [item for index, item in enumerate(conversations) if index in validation_indices]
    train_ids = _encode_many(train, tokenizer)
    validation_ids = _encode_many(validation, tokenizer)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_count, train_sha = _write_u32le(destination / "train.bin", train_ids)
    val_count, val_sha = _write_u32le(destination / "val.bin", validation_ids)
    tokenizer.save(destination / "tokenizer.json")

    input_bytes = Path(input_path).read_bytes()
    manifest: dict[str, Any] = {
        "format": "zika-token-ids-u32le-v1",
        "tokenizer": "char-v0.1",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "train": {
            "conversations": len(train), "tokens": train_count,
            "file": "train.bin", "sha256": train_sha,
        },
        "validation": {
            "conversations": len(validation), "tokens": val_count,
            "file": "val.bin", "sha256": val_sha,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara o corpus de treino do ZIKA-AI.")
    parser.add_argument("--input", default="data/dialogos.jsonl", help="corpus JSONL de origem")
    parser.add_argument("--output-dir", default="data/processed", help="pasta para os artefactos")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        manifest = prepare_dataset(
            args.input, args.output_dir,
            validation_ratio=args.validation_ratio, seed=args.seed,
        )
    except (DatasetError, OSError, TokenizerError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    print("Corpus preparado com sucesso")
    print(
        f"  treino:     {manifest['train']['conversations']} conversas, "
        f"{manifest['train']['tokens']} tokens"
    )
    print(
        f"  validacao:  {manifest['validation']['conversations']} conversas, "
        f"{manifest['validation']['tokens']} tokens"
    )
    print(f"  vocabulario: {manifest['vocab_size']} tokens")
    print(f"  saida:       {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

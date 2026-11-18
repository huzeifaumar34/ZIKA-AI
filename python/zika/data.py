"""Validacao, deduplicacao e preparacao de corpus JSONL para o ZIKA-AI.

Cada linha representa uma conversa completa. As conversas sao separadas entre
treino, validacao e teste antes da serializacao, evitando fuga entre conjuntos.
Os artefactos binarios usam IDs uint32 little-endian.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

from .tokenizer import BpeTokenizer, CharTokenizer, SPECIAL_TOKENS, TokenizerError

ROLE_TOKEN = {"user": "<|user|>", "ia": "<|ia|>"}


class DatasetError(ValueError):
    """Corpus JSONL invalido ou nao representavel."""


def validate_conversation(
    item: Any, *, source: str = "<memory>", line_number: int = 1,
    tokenizer: CharTokenizer | BpeTokenizer | None = None,
    validate_characters: bool = True,
) -> dict[str, list[dict[str, str]]]:
    if tokenizer is None and validate_characters:
        tokenizer = CharTokenizer()
    if not isinstance(item, dict) or set(item) != {"turns"}:
        raise DatasetError(
            f"{source}:{line_number}: cada registo deve conter apenas a chave 'turns'"
        )
    turns = item["turns"]
    if not isinstance(turns, list) or len(turns) < 2:
        raise DatasetError(
            f"{source}:{line_number}: 'turns' tem de conter pelo menos duas mensagens"
        )

    clean_turns: list[dict[str, str]] = []
    expected_role = "user"
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or set(turn) != {"role", "text"}:
            raise DatasetError(
                f"{source}:{line_number}: turno {index + 1} deve conter 'role' e 'text'"
            )
        role, text = turn["role"], turn["text"]
        if not isinstance(role, str) or role not in ROLE_TOKEN:
            raise DatasetError(
                f"{source}:{line_number}: role invalido no turno {index + 1}: {role!r}"
            )
        if role != expected_role:
            raise DatasetError(
                f"{source}:{line_number}: esperado role {expected_role!r} no turno {index + 1}"
            )
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(
                f"{source}:{line_number}: texto vazio ou invalido no turno {index + 1}"
            )
        if any(marker in text for marker in SPECIAL_TOKENS):
            raise DatasetError(
                f"{source}:{line_number}: texto nao pode conter marcadores especiais reservados"
            )
        if validate_characters and tokenizer is not None:
            try:
                tokenizer.encode(text, parse_special_tokens=False)
            except TokenizerError as exc:
                raise DatasetError(f"{source}:{line_number}: {exc}") from exc
        clean_turns.append({"role": role, "text": text})
        expected_role = "ia" if expected_role == "user" else "user"

    if clean_turns[-1]["role"] != "ia":
        raise DatasetError(
            f"{source}:{line_number}: a conversa tem de terminar com uma resposta da ia"
        )
    return {"turns": clean_turns}


def serialize_conversation(conversation: dict[str, list[dict[str, str]]]) -> str:
    pieces: list[str] = []
    for turn in conversation["turns"]:
        pieces.extend((ROLE_TOKEN[turn["role"]], turn["text"]))
    pieces.append("<|fim|>")
    return "".join(pieces)


def _fingerprint(conversation: dict[str, list[dict[str, str]]]) -> str:
    serialized = serialize_conversation(conversation)
    normalized = " ".join(
        unicodedata.normalize("NFKC", serialized).casefold().split()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _source_paths(path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(item) for item in path]
    if not paths:
        raise DatasetError("indique pelo menos um ficheiro de corpus")
    return paths


def _load_sources(
    paths: Sequence[Path], tokenizer: CharTokenizer | BpeTokenizer | None
) -> tuple[list[dict[str, list[dict[str, str]]]], int]:
    conversations: list[dict[str, list[dict[str, str]]]] = []
    seen: set[str] = set()
    duplicates = 0
    for source in paths:
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DatasetError(f"nao foi possivel ler {source}: {exc}") from exc
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"{source}:{line_number}: JSON invalido na coluna {exc.colno}: {exc.msg}"
                ) from exc
            conversation = validate_conversation(
                item, source=str(source), line_number=line_number,
                tokenizer=tokenizer, validate_characters=tokenizer is not None,
            )
            fingerprint = _fingerprint(conversation)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            conversations.append(conversation)
    return conversations, duplicates


def load_jsonl(
    path: str | Path | Sequence[str | Path],
    tokenizer: CharTokenizer | BpeTokenizer | None = None,
) -> list[dict[str, list[dict[str, str]]]]:
    conversations, _ = _load_sources(_source_paths(path), tokenizer or CharTokenizer())
    return conversations


def encode_conversation(
    conversation: dict[str, list[dict[str, str]]],
    tokenizer: CharTokenizer | BpeTokenizer | None = None
) -> list[int]:
    tokenizer = tokenizer or CharTokenizer()
    return tokenizer.encode(serialize_conversation(conversation))


def _encode_many(
    conversations: Sequence[dict[str, list[dict[str, str]]]],
    tokenizer: CharTokenizer | BpeTokenizer
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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return count, hashlib.sha256(payload).hexdigest()


def prepare_dataset(
    input_path: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    tokenizer_kind: str = "char",
    bpe_vocab_size: int = 512,
) -> dict[str, Any]:
    if not 0.0 < validation_ratio < 1.0:
        raise DatasetError("validation_ratio tem de estar entre 0 e 1")
    if not 0.0 < test_ratio < 1.0:
        raise DatasetError("test_ratio tem de estar entre 0 e 1")
    if validation_ratio + test_ratio >= 1.0:
        raise DatasetError("validation_ratio + test_ratio tem de ser inferior a 1")

    if tokenizer_kind not in ("char", "bpe"):
        raise DatasetError("tokenizer_kind tem de ser 'char' ou 'bpe'")
    if tokenizer_kind == "char":
        validation_tokenizer: CharTokenizer | BpeTokenizer | None = CharTokenizer()
    else:
        validation_tokenizer = None
    sources = _source_paths(input_path)
    conversations, duplicate_count = _load_sources(sources, validation_tokenizer)
    if len(conversations) < 3:
        raise DatasetError(
            "sao necessarias pelo menos tres conversas unicas para treino, validacao e teste"
        )

    indices = list(range(len(conversations)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, int(round(len(indices) * validation_ratio)))
    test_count = max(1, int(round(len(indices) * test_ratio)))
    if validation_count + test_count >= len(indices):
        raise DatasetError("o corpus e demasiado pequeno para os tres conjuntos")
    validation_indices = set(indices[:validation_count])
    test_indices = set(indices[validation_count:validation_count + test_count])

    train = [item for i, item in enumerate(conversations)
             if i not in validation_indices and i not in test_indices]
    validation = [item for i, item in enumerate(conversations)
                  if i in validation_indices]
    test = [item for i, item in enumerate(conversations) if i in test_indices]
    if tokenizer_kind == "char":
        tokenizer: CharTokenizer | BpeTokenizer = CharTokenizer()
    else:
        tokenizer = BpeTokenizer.train(
            (serialize_conversation(item) for item in train),
            vocab_size=bpe_vocab_size,
        )
    train_ids = _encode_many(train, tokenizer)
    validation_ids = _encode_many(validation, tokenizer)
    test_ids = _encode_many(test, tokenizer)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_count, train_sha = _write_u32le(destination / "train.bin", train_ids)
    val_count, val_sha = _write_u32le(destination / "val.bin", validation_ids)
    test_token_count, test_sha = _write_u32le(destination / "test.bin", test_ids)
    tokenizer.save(destination / "tokenizer.json")
    if isinstance(tokenizer, BpeTokenizer):
        tokenizer.save_binary(destination / "tokenizer.zkt")

    source_info = []
    for source in sources:
        raw = source.read_bytes()
        source_info.append({
            "path": str(source),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    manifest: dict[str, Any] = {
        "format": "zika-token-ids-u32le-v1",
        "schema_version": 2,
        "tokenizer": "char-v0.1" if tokenizer_kind == "char" else "bpe-byte-v0.1",
        "tokenizer_kind": tokenizer_kind,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "sources": source_info,
        "unique_conversations": len(conversations),
        "duplicates_removed": duplicate_count,
        "train": {
            "conversations": len(train), "tokens": train_count,
            "file": "train.bin", "sha256": train_sha,
        },
        "validation": {
            "conversations": len(validation), "tokens": val_count,
            "file": "val.bin", "sha256": val_sha,
        },
        "test": {
            "conversations": len(test), "tokens": test_token_count,
            "file": "test.bin", "sha256": test_sha,
        },
    }
    temporary_manifest = destination / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(destination / "manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara o corpus do ZIKA-AI.")
    parser.add_argument(
        "--input", nargs="+",
        default=["data/dialogos.jsonl"],
        help="um ou mais ficheiros JSONL de origem",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--vocab-size", type=int, default=512,
                        help="tamanho alvo do vocabulario BPE (259 tokens base)")
    args = parser.parse_args(argv)

    try:
        manifest = prepare_dataset(
            args.input, args.output_dir,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            tokenizer_kind=args.tokenizer,
            bpe_vocab_size=args.vocab_size,
        )
    except (DatasetError, OSError, TokenizerError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    print("Corpus preparado com sucesso")
    for key in ("train", "validation", "test"):
        section = manifest[key]
        print(
            f"  {key:11s} {section['conversations']:5d} conversas, "
            f"{section['tokens']:7d} tokens"
        )
    print(
        f"  unicas: {manifest['unique_conversations']} | "
        f"duplicadas removidas: {manifest['duplicates_removed']}"
    )
    print(f"  vocabulario: {manifest['vocab_size']} tokens")
    print(f"  saida:       {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

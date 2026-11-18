"""Tokenizers char-v0.1 e BPE byte-v0.1 do ZIKA-AI.

A ordem do vocabulario e parte do contrato com core/src/format.cpp:
3 tokens especiais, newline, ASCII imprimivel e 25 caracteres acentuados.
Nao ha normalizacao Unicode silenciosa: a codificacao e reversivel.
"""
from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

SPECIAL_TOKENS = ("<|user|>", "<|ia|>", "<|fim|>")
ACCENTED_CHARS = tuple("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÇ")
CHAR_VOCAB = (
    *SPECIAL_TOKENS,
    "\n",
    *(chr(codepoint) for codepoint in range(32, 127)),
    *ACCENTED_CHARS,
)
TOKENIZER_KIND = "char"
TOKENIZER_VERSION = "char-v0.1"


class TokenizerError(ValueError):
    """Texto ou sequencia de IDs nao representavel pelo tokenizer."""


class CharTokenizer:
    """Codifica texto sem perdas com o vocabulario fixo de 124 caracteres."""

    def __init__(self, tokens: Sequence[str] = CHAR_VOCAB) -> None:
        self.tokens = tuple(tokens)
        if self.tokens != CHAR_VOCAB:
            raise TokenizerError(
                "o vocabulario char-v0.1 difere da ordem canonica usada pelo nucleo C++"
            )
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self._special_ids = {token: self.token_to_id[token] for token in SPECIAL_TOKENS}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, *, parse_special_tokens: bool = True) -> list[int]:
        """Converte texto para IDs; caracteres desconhecidos geram erro explicito.

        Com parse_special_tokens=True, os marcadores reservados sao tokens atomicos.
        Para texto literal de utilizador use False.
        """
        if not isinstance(text, str):
            raise TypeError("text tem de ser str")
        ids: list[int] = []
        index = 0
        while index < len(text):
            matched = None
            if parse_special_tokens:
                for marker in SPECIAL_TOKENS:
                    if text.startswith(marker, index):
                        matched = marker
                        break
            if matched is not None:
                ids.append(self._special_ids[matched])
                index += len(matched)
                continue
            char = text[index]
            token_id = self.token_to_id.get(char)
            if token_id is None:
                raise TokenizerError(
                    f"caractere nao suportado U+{ord(char):04X} na posicao {index}: {char!r}"
                )
            ids.append(token_id)
            index += 1
        return ids

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = False
    ) -> str:
        """Reconstrui exatamente a string original a partir dos IDs."""
        parts: list[str] = []
        for position, raw_id in enumerate(token_ids):
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TokenizerError(f"ID nao inteiro na posicao {position}")
            if raw_id < 0 or raw_id >= self.vocab_size:
                raise TokenizerError(f"ID fora do vocabulario na posicao {position}: {raw_id}")
            token = self.tokens[raw_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            parts.append(token)
        return "".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": TOKENIZER_KIND,
            "version": TOKENIZER_VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "tokens": list(self.tokens),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"nao foi possivel ler tokenizer {source}: {exc}") from exc
        if not isinstance(data, dict) or data.get("kind") != TOKENIZER_KIND:
            raise TokenizerError("ficheiro nao descreve um tokenizer char")
        if data.get("version") != TOKENIZER_VERSION:
            raise TokenizerError("versao do tokenizer nao suportada")
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
            raise TokenizerError("lista de tokens invalida")
        tokenizer = cls(tokens)
        if data.get("vocab_size") != tokenizer.vocab_size:
            raise TokenizerError("vocab_size nao corresponde a lista de tokens")
        return tokenizer


# BPE byte-level v0.1. Three structural tokens are followed by all 256
# byte values. This gives lossless UTF-8 coverage, including characters not
# seen during tokenizer training, while merges are learned only from train data.
BPE_KIND = "bpe"
BPE_VERSION = "bpe-byte-v0.1"
BPE_BASE_VOCAB_SIZE = 3 + 256
BPE_BINARY_MAGIC = b"ZTKZ"
BPE_BINARY_VERSION = 1
MAX_BPE_TOKEN_BYTES = 64


class BpeTokenizer:
    """Byte-level BPE trained with deterministic, ranked pair merges.

    IDs 0..2 are the ZIKA conversation markers; IDs 3..258 represent bytes
    0..255; each merge adds exactly one token at the end of the vocabulary.
    """

    kind = BPE_KIND
    version = BPE_VERSION

    def __init__(
        self,
        tokens: Sequence[bytes],
        merges: Sequence[tuple[int, int]],
    ) -> None:
        self.tokens = tuple(bytes(token) for token in tokens)
        self.merges = tuple((int(left), int(right)) for left, right in merges)
        self._validate()
        self._special_ids = {
            marker: index for index, marker in enumerate(SPECIAL_TOKENS)
        }

    @classmethod
    def _base_tokens(cls) -> list[bytes]:
        return [
            *(marker.encode("utf-8") for marker in SPECIAL_TOKENS),
            *(bytes((value,)) for value in range(256)),
        ]

    @staticmethod
    def _raw_encode(text: str, *, parse_special_tokens: bool) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text tem de ser str")
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            matched = None
            if parse_special_tokens:
                for marker_id, marker in enumerate(SPECIAL_TOKENS):
                    if text.startswith(marker, pos):
                        matched = (marker_id, marker)
                        break
            if matched is not None:
                marker_id, marker = matched
                ids.append(marker_id)
                pos += len(marker)
                continue

            next_marker = len(text)
            if parse_special_tokens:
                positions = [
                    found for marker in SPECIAL_TOKENS
                    if (found := text.find(marker, pos)) >= 0
                ]
                if positions:
                    next_marker = min(positions)
            chunk = text[pos:next_marker]
            ids.extend(byte + 3 for byte in chunk.encode("utf-8"))
            pos = next_marker
        return ids

    @staticmethod
    def _replace_pair(
        sequence: Sequence[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        result: list[int] = []
        index = 0
        left, right = pair
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == left
                and sequence[index + 1] == right
            ):
                result.append(new_id)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "BpeTokenizer":
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TokenizerError("vocab_size tem de ser inteiro")
        if vocab_size < BPE_BASE_VOCAB_SIZE or vocab_size > 65536:
            raise TokenizerError(
                f"vocab_size BPE tem de estar entre {BPE_BASE_VOCAB_SIZE} e 65536"
            )
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
            raise TokenizerError("min_frequency tem de ser inteiro")
        if min_frequency < 2:
            raise TokenizerError("min_frequency tem de ser pelo menos 2")

        sequences = [
            cls._raw_encode(text, parse_special_tokens=True) for text in texts
        ]
        if not sequences or not any(sequences):
            raise TokenizerError("nao existem textos para treinar o tokenizer BPE")

        tokens = cls._base_tokens()
        merges: list[tuple[int, int]] = []
        while len(tokens) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                for left, right in zip(sequence, sequence[1:]):
                    # Never merge structural markers into ordinary text.
                    if left < 3 or right < 3:
                        continue
                    if len(tokens[left]) + len(tokens[right]) > MAX_BPE_TOKEN_BYTES:
                        continue
                    counts[(left, right)] += 1
            if not counts:
                break
            best_pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
            )
            if frequency < min_frequency:
                break

            new_id = len(tokens)
            tokens.append(tokens[best_pair[0]] + tokens[best_pair[1]])
            merges.append(best_pair)
            sequences = [
                cls._replace_pair(sequence, best_pair, new_id)
                for sequence in sequences
            ]

        return cls(tokens, merges)

    def _validate(self) -> None:
        if len(self.tokens) < BPE_BASE_VOCAB_SIZE:
            raise TokenizerError("vocabulario BPE nao contem os 256 tokens de byte")
        expected_special = tuple(marker.encode("utf-8") for marker in SPECIAL_TOKENS)
        if self.tokens[:3] != expected_special:
            raise TokenizerError("os tres tokens especiais BPE estao fora de ordem")
        for value in range(256):
            if self.tokens[value + 3] != bytes((value,)):
                raise TokenizerError("tokens base BPE nao correspondem a ordem byte-v0")
        if len(self.merges) != len(self.tokens) - BPE_BASE_VOCAB_SIZE:
            raise TokenizerError("numero de merges BPE difere do tamanho do vocabulario")
        for index, (left, right) in enumerate(self.merges):
            new_id = BPE_BASE_VOCAB_SIZE + index
            if left < 3 or right < 3 or left >= new_id or right >= new_id:
                raise TokenizerError(f"merge BPE invalido na posicao {index}")
            expected = self.tokens[left] + self.tokens[right]
            if len(expected) > MAX_BPE_TOKEN_BYTES or self.tokens[new_id] != expected:
                raise TokenizerError(f"token BPE nao corresponde ao merge {index}")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def _apply_merges(self, ids: Sequence[int]) -> list[int]:
        result = list(ids)
        for rank, pair in enumerate(self.merges):
            result = self._replace_pair(
                result, pair, BPE_BASE_VOCAB_SIZE + rank
            )
        return result

    def encode(self, text: str, *, parse_special_tokens: bool = True) -> list[int]:
        """Encode UTF-8 bytes losslessly; unknown Unicode needs no UNK token."""
        raw_ids = self._raw_encode(text, parse_special_tokens=parse_special_tokens)
        return self._apply_merges(raw_ids)

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = False
    ) -> str:
        output = bytearray()
        for position, raw_id in enumerate(token_ids):
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise TokenizerError(f"ID nao inteiro na posicao {position}")
            if raw_id < 0 or raw_id >= self.vocab_size:
                raise TokenizerError(
                    f"ID fora do vocabulario na posicao {position}: {raw_id}"
                )
            if skip_special_tokens and raw_id < 3:
                continue
            output.extend(self.tokens[raw_id])
        try:
            return output.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TokenizerError(f"sequencia BPE nao e UTF-8 valida: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": BPE_KIND,
            "version": BPE_VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "byte_fallback": True,
            "tokens_hex": [token.hex() for token in self.tokens],
            "merges": [[left, right] for left, right in self.merges],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            tokens_hex = data["tokens_hex"]
            merges_raw = data["merges"]
            if not isinstance(tokens_hex, list) or not all(
                isinstance(item, str) for item in tokens_hex
            ):
                raise TokenizerError("tokens_hex invalido")
            if not isinstance(merges_raw, list):
                raise TokenizerError("merges invalido")
            tokenizer = cls(
                [bytes.fromhex(item) for item in tokens_hex],
                [(int(pair[0]), int(pair[1])) for pair in merges_raw],
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TokenizerError):
                raise
            raise TokenizerError(f"nao foi possivel ler tokenizer BPE {source}: {exc}") from exc
        if data.get("kind") != BPE_KIND or data.get("version") != BPE_VERSION:
            raise TokenizerError("tipo ou versao do tokenizer BPE nao suportados")
        if data.get("vocab_size") != tokenizer.vocab_size:
            raise TokenizerError("vocab_size nao corresponde aos tokens BPE")
        return tokenizer

    def to_binary(self) -> bytes:
        self._validate()
        payload = bytearray(BPE_BINARY_MAGIC)
        payload.extend(struct.pack(
            "<IIII", BPE_BINARY_VERSION, 1, self.vocab_size, len(self.merges)
        ))
        for token in self.tokens:
            payload.extend(struct.pack("<I", len(token)))
            payload.extend(token)
        for left, right in self.merges:
            payload.extend(struct.pack("<II", left, right))
        checksum = _crc32(payload)
        payload.extend(struct.pack("<I", checksum))
        return bytes(payload)

    def save_binary(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(self.to_binary())
        temporary.replace(destination)


def _crc32(data: bytes | bytearray) -> int:
    value = 0xFFFFFFFF
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0xEDB88320 if value & 1 else 0)
    return value ^ 0xFFFFFFFF

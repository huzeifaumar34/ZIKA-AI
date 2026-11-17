"""Tokenizer v0.1 do ZIKA-AI: um token por caractere Unicode suportado.

A ordem do vocabulario e parte do contrato com core/src/format.cpp:
3 tokens especiais, newline, ASCII imprimivel e 25 caracteres acentuados.
Nao ha normalizacao Unicode silenciosa: a codificacao e reversivel.
"""
from __future__ import annotations

import json
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

import json
import tempfile
import unittest
from pathlib import Path

from zika.tokenizer import (
    ACCENTED_CHARS,
    CHAR_VOCAB,
    SPECIAL_TOKENS,
    CharTokenizer,
    BpeTokenizer,
    BPE_BASE_VOCAB_SIZE,
    TokenizerError,
)


class CharTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer()

    def test_vocabulary_matches_cpp_contract(self):
        self.assertEqual(len(ACCENTED_CHARS), 25)
        self.assertEqual(len(CHAR_VOCAB), 124)
        self.assertEqual(self.tokenizer.vocab_size, 124)
        self.assertEqual(self.tokenizer.tokens[:3], SPECIAL_TOKENS)
        self.assertEqual(self.tokenizer.tokens[3], "\n")

    def test_roundtrip_ascii_and_portuguese(self):
        text = "Ola! A ZIKA aprende portugues: acao, informacao, coracao, aviao."
        text += " Acentos: áàâã éê í óôõ úü ç ÁÀÂÃ ÉÊ Í ÓÔÕ Ú Ç."
        ids = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_special_tokens_are_atomic(self):
        text = "<|user|>Ola<|ia|>Bom dia!<|fim|>"
        ids = self.tokenizer.encode(text)
        self.assertEqual(ids[0:1], [0])
        self.assertIn(1, ids)
        self.assertEqual(ids[-1], 2)
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual(
            self.tokenizer.decode(ids, skip_special_tokens=True),
            "OlaBom dia!",
        )

    def test_literal_special_token_can_be_encoded_as_characters(self):
        text = "<|user|>"
        ids = self.tokenizer.encode(text, parse_special_tokens=False)
        self.assertGreater(len(ids), 1)
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_unknown_character_is_not_silently_dropped(self):
        with self.assertRaisesRegex(TokenizerError, "U\\+1F30D"):
            self.tokenizer.encode("ola 🌍")

    def test_invalid_ids_are_rejected(self):
        for values in ([-1], [124], [True], ["1"]):
            with self.subTest(values=values), self.assertRaises(TokenizerError):
                self.tokenizer.decode(values)

    def test_vocab_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = CharTokenizer.load(path)
            self.assertEqual(loaded.tokens, self.tokenizer.tokens)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["vocab_size"], 124
            )


class BpeTokenizerTests(unittest.TestCase):
    def test_deterministic_training_and_utf8_byte_fallback(self):
        corpus = [
            "ola mundo ola mundo",
            "ola mundo ola mundo",
            "programar com cuidado programar com cuidado",
        ]
        first = BpeTokenizer.train(corpus, vocab_size=280, min_frequency=2)
        second = BpeTokenizer.train(corpus, vocab_size=280, min_frequency=2)
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(first.merges, second.merges)
        self.assertGreater(first.vocab_size, BPE_BASE_VOCAB_SIZE)

        text = "Olá, mundo! 🙂 中文 — texto desconhecido."
        ids = first.encode(text, parse_special_tokens=False)
        self.assertLess(len(ids), len(text.encode("utf-8")))
        self.assertEqual(first.decode(ids), text)

    def test_special_tokens_are_reserved_but_user_text_can_be_literal(self):
        tokenizer = BpeTokenizer.train(
            ["<|user|>ola<|ia|>bom dia<|fim|>"] * 3,
            vocab_size=270,
        )
        serialized_ids = tokenizer.encode("<|user|>ola<|ia|>")
        self.assertEqual(serialized_ids[0], 0)
        self.assertIn(1, serialized_ids)
        literal = "<|user|>"
        literal_ids = tokenizer.encode(literal, parse_special_tokens=False)
        self.assertTrue(all(token_id >= 3 for token_id in literal_ids))
        self.assertEqual(tokenizer.decode(literal_ids), literal)
        self.assertEqual(
            tokenizer.decode(serialized_ids, skip_special_tokens=True),
            "ola",
        )

    def test_json_and_binary_roundtrip_with_checksum(self):
        tokenizer = BpeTokenizer.train(["abc abc abc", "abc abc abc"], vocab_size=280)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "tokenizer.json"
            binary_path = Path(tmp) / "tokenizer.zkt"
            tokenizer.save(json_path)
            tokenizer.save_binary(binary_path)
            loaded = BpeTokenizer.load(json_path)
            self.assertEqual(loaded.tokens, tokenizer.tokens)
            self.assertEqual(loaded.merges, tokenizer.merges)
            self.assertEqual(loaded.encode("🙂abc"), tokenizer.encode("🙂abc"))
            raw = binary_path.read_bytes()
            self.assertEqual(raw[:4], b"ZTKZ")
            self.assertEqual(len(raw), len(tokenizer.to_binary()))
            self.assertEqual(int.from_bytes(raw[-4:], "little"),
                             __import__("zlib").crc32(raw[:-4]))

    def test_invalid_vocab_and_ids_are_rejected(self):
        with self.assertRaises(TokenizerError):
            BpeTokenizer.train(["abc"], vocab_size=BPE_BASE_VOCAB_SIZE - 1)
        tokenizer = BpeTokenizer.train(["abc abc"] * 2, vocab_size=270)
        with self.assertRaises(TokenizerError):
            tokenizer.decode([tokenizer.vocab_size])
        with self.assertRaises(TokenizerError):
            tokenizer.decode([True])


if __name__ == "__main__":
    unittest.main()

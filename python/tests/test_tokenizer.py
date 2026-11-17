import json
import tempfile
import unittest
from pathlib import Path

from zika.tokenizer import (
    ACCENTED_CHARS,
    CHAR_VOCAB,
    SPECIAL_TOKENS,
    CharTokenizer,
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


if __name__ == "__main__":
    unittest.main()

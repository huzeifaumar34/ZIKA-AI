import json
import struct
import tempfile
import unittest
from pathlib import Path

from zika.data import (
    DatasetError,
    encode_conversation,
    load_jsonl,
    prepare_dataset,
    serialize_conversation,
    validate_conversation,
)
from zika.tokenizer import CharTokenizer


SAMPLE = {
    "turns": [
        {"role": "user", "text": "Ola!"},
        {"role": "ia", "text": "Ola. Como posso ajudar?"},
        {"role": "user", "text": "Explica a ZIKA."},
        {"role": "ia", "text": "E um modelo pequeno de linguagem."},
    ]
}


class DatasetTests(unittest.TestCase):
    def test_conversation_serialization_and_roundtrip(self):
        tokenizer = CharTokenizer()
        serialized = serialize_conversation(SAMPLE)
        self.assertEqual(
            serialized,
            "<|user|>Ola!<|ia|>Ola. Como posso ajudar?"
            "<|user|>Explica a ZIKA.<|ia|>E um modelo pequeno de linguagem.<|fim|>",
        )
        ids = encode_conversation(SAMPLE, tokenizer)
        self.assertEqual(tokenizer.decode(ids), serialized)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 2)

    def test_rejects_bad_roles_and_unsupported_characters(self):
        bad_role = {"turns": [{"role": "ia", "text": "Ola"}, {"role": "user", "text": "Oi"}]}
        with self.assertRaises(DatasetError):
            validate_conversation(bad_role)
        bad_text = {"turns": [{"role": "user", "text": "Ola 🌍"}, {"role": "ia", "text": "Oi"}]}
        with self.assertRaises(DatasetError):
            validate_conversation(bad_text)

    def test_rejects_unclosed_conversation(self):
        bad = {"turns": [{"role": "user", "text": "Ola"}, {"role": "ia", "text": "Oi"},
                         {"role": "user", "text": "Mais uma pergunta"}]}
        with self.assertRaisesRegex(DatasetError, "terminar"):
            validate_conversation(bad)

    def test_jsonl_load_and_line_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(json.dumps(SAMPLE, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(len(load_jsonl(path)), 1)
            path.write_text("{nao-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, ":1: JSON invalido"):
                load_jsonl(path)

    def test_prepare_is_deterministic_and_writes_uint32le(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dialogos.jsonl"
            rows = [SAMPLE, {
                "turns": [
                    {"role": "user", "text": "O que e energia solar?"},
                    {"role": "ia", "text": "E energia obtida da luz do sol."},
                ]
            }, {
                "turns": [
                    {"role": "user", "text": "Como estudar melhor?"},
                    {"role": "ia", "text": "Divide o estudo em blocos curtos."},
                ]
            }]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            out_a, out_b = root / "a", root / "b"
            manifest_a = prepare_dataset(source, out_a, seed=7)
            manifest_b = prepare_dataset(source, out_b, seed=7)
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["train"]["conversations"], 2)
            self.assertEqual(manifest_a["validation"]["conversations"], 1)
            train_bytes = (out_a / "train.bin").read_bytes()
            self.assertEqual(len(train_bytes) % 4, 0)
            first_id = struct.unpack("<I", train_bytes[:4])[0]
            self.assertEqual(first_id, 0)
            self.assertEqual(
                (out_a / "train.bin").read_bytes(),
                (out_b / "train.bin").read_bytes(),
            )
            self.assertTrue((out_a / "manifest.json").exists())
            self.assertTrue((out_a / "tokenizer.json").exists())

    def test_invalid_validation_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.jsonl"
            path.write_text(
                json.dumps(SAMPLE) + "\n" + json.dumps(SAMPLE) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError):
                prepare_dataset(path, Path(tmp) / "out", validation_ratio=1.0)


if __name__ == "__main__":
    unittest.main()

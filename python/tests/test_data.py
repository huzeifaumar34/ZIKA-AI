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
from zika.tokenizer import BpeTokenizer, CharTokenizer


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

    def test_duplicate_conversations_are_removed_across_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.jsonl", root / "b.jsonl"
            row = json.dumps(SAMPLE, ensure_ascii=False) + "\n"
            a.write_text(row, encoding="utf-8")
            b.write_text(row.replace("Ola!", "OLA!"), encoding="utf-8")
            # Exact semantic normalisation is case-insensitive and stable.
            self.assertEqual(len(load_jsonl([a, b])), 1)

    def test_three_way_split_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dialogos.jsonl"
            rows = [
                SAMPLE,
                {"turns": [{"role": "user", "text": "O que e energia solar?"},
                           {"role": "ia", "text": "E energia obtida da luz do sol."}]},
                {"turns": [{"role": "user", "text": "Como estudar melhor?"},
                           {"role": "ia", "text": "Divide o estudo em blocos curtos."}]},
                {"turns": [{"role": "user", "text": "O que e memoria RAM?"},
                           {"role": "ia", "text": "E memoria temporaria usada pelos programas."}]},
                {"turns": [{"role": "user", "text": "Para que serve o Git?"},
                           {"role": "ia", "text": "Regista alteracoes no projeto."}]},
                {"turns": [{"role": "user", "text": "O que e uma API?"},
                           {"role": "ia", "text": "Define como programas comunicam."}]},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            out_a, out_b = root / "a", root / "b"
            manifest_a = prepare_dataset(source, out_a, seed=7)
            manifest_b = prepare_dataset(source, out_b, seed=7)
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["train"]["conversations"], 4)
            self.assertEqual(manifest_a["validation"]["conversations"], 1)
            self.assertEqual(manifest_a["test"]["conversations"], 1)
            for name in ("train.bin", "val.bin", "test.bin"):
                raw_a = (out_a / name).read_bytes()
                raw_b = (out_b / name).read_bytes()
                self.assertEqual(raw_a, raw_b)
                self.assertGreater(len(raw_a), 0)
                self.assertEqual(len(raw_a) % 4, 0)
                self.assertEqual(struct.unpack("<I", raw_a[:4])[0], 0)
            self.assertTrue((out_a / "manifest.json").exists())
            self.assertTrue((out_a / "tokenizer.json").exists())

    def test_invalid_split_ratios_and_too_small_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.jsonl"
            path.write_text(
                "".join(json.dumps(SAMPLE) + "\n" for _ in range(3)),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetError):
                prepare_dataset(path, root / "out", validation_ratio=0.7)
            path.write_text(json.dumps(SAMPLE) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                prepare_dataset(path, root / "out")

    def test_bpe_pipeline_handles_unseen_unicode_and_writes_binary_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "unicode.jsonl"
            rows = [
                {"turns": [{"role": "user", "text": "Como funciona o sol?"},
                           {"role": "ia", "text": "O sol e uma estrela que emite luz."}]},
                {"turns": [{"role": "user", "text": "O que e um oceano?"},
                           {"role": "ia", "text": "E uma grande massa de agua salgada."}]},
                {"turns": [{"role": "user", "text": "O que significa emoji?"},
                           {"role": "ia", "text": "E um simbolo visual usado em mensagens 🙂."}]},
                {"turns": [{"role": "user", "text": "Como se diz ola em japones?"},
                           {"role": "ia", "text": "Uma forma comum e konnichiwa: こんにちは."}]},
                {"turns": [{"role": "user", "text": "O que e uma rede?"},
                           {"role": "ia", "text": "Liga dispositivos para trocar informacao."}]},
                {"turns": [{"role": "user", "text": "O que e uma biblioteca?"},
                           {"role": "ia", "text": "E uma colecao organizada de recursos."}]},
                {"turns": [{"role": "user", "text": "O que e um planeta?"},
                           {"role": "ia", "text": "E um corpo celeste que orbita uma estrela."}]},
                {"turns": [{"role": "user", "text": "Como se escreve obrigado em grego?"},
                           {"role": "ia", "text": "Uma expressao e ευχαριστώ."}]},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows),
                encoding="utf-8",
            )
            out = root / "processed"
            manifest = prepare_dataset(
                source, out, seed=19, tokenizer_kind="bpe", bpe_vocab_size=300,
                validation_ratio=0.125, test_ratio=0.125,
            )
            self.assertEqual(manifest["tokenizer"], "bpe-byte-v0.1")
            self.assertEqual(manifest["tokenizer_kind"], "bpe")
            self.assertTrue((out / "tokenizer.zkt").exists())
            self.assertEqual((out / "tokenizer.zkt").read_bytes()[:4], b"ZTKZ")
            tokenizer = BpeTokenizer.load(out / "tokenizer.json")
            self.assertEqual(tokenizer.vocab_size, manifest["vocab_size"])
            novel = "🙂 こんにちは ευχαριστώ"
            self.assertEqual(
                tokenizer.decode(tokenizer.encode(novel, parse_special_tokens=False)),
                novel,
            )
            for file_name in ("train.bin", "val.bin", "test.bin"):
                self.assertEqual((out / file_name).stat().st_size % 4, 0)


if __name__ == "__main__":
    unittest.main()

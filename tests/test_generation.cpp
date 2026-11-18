#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "zika/generation.hpp"
#include "zika/transformer.hpp"

using namespace zika;

namespace {
int checks = 0;
int failures = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; std::fprintf(stderr, "FALHOU %s:%d: %s\n", __FILE__, __LINE__, #condition); } } while (0)

void put_u32(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int i = 0; i < 4; ++i)
    bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

Tokenizer tiny_bpe_tokenizer() {
  Tokenizer tok;
  tok.kind = TokenizerKind::BPE;
  tok.tokens = {"<|user|>", "<|ia|>", "<|fim|>"};
  for (int byte = 0; byte < 256; ++byte)
    tok.tokens.emplace_back(1, static_cast<char>(byte));
  tok.tokens.push_back("ab"); // ID 259
  tok.merges.emplace_back(3u + static_cast<uint32_t>('a'),
                          3u + static_cast<uint32_t>('b'));
  return tok;
}

Model tiny_char_model() {
  Config cfg;
  cfg.vocab_size = 5;
  cfg.n_layers = 1;
  cfg.d_model = 4;
  cfg.n_heads = 2;
  cfg.d_ff = 8;
  cfg.ctx_len = 8;
  Tokenizer tok;
  tok.kind = TokenizerKind::Char;
  tok.tokens = {"<|user|>", "<|ia|>", "<|fim|>", "a", "b"};
  Model model = make_empty_model(cfg, tok);
  initialize_weights(model, 123);
  return model;
}

Model tiny_bpe_model() {
  Tokenizer tok = tiny_bpe_tokenizer();
  Config cfg;
  cfg.vocab_size = static_cast<uint32_t>(tok.tokens.size());
  cfg.n_layers = 1;
  cfg.d_model = 4;
  cfg.n_heads = 2;
  cfg.d_ff = 8;
  cfg.ctx_len = 8;
  Model model = make_empty_model(cfg, tok);
  initialize_weights(model, 456);
  return model;
}

std::string write_bpe_fixture(const std::string &path, const Tokenizer &tok) {
  std::vector<uint8_t> bytes{'Z', 'T', 'K', 'Z'};
  put_u32(bytes, 1); // ZTKZ version
  put_u32(bytes, 1); // BPE
  put_u32(bytes, static_cast<uint32_t>(tok.tokens.size()));
  put_u32(bytes, static_cast<uint32_t>(tok.merges.size()));
  for (const std::string &token : tok.tokens) {
    put_u32(bytes, static_cast<uint32_t>(token.size()));
    bytes.insert(bytes.end(), token.begin(), token.end());
  }
  for (const auto &merge : tok.merges) {
    put_u32(bytes, merge.first);
    put_u32(bytes, merge.second);
  }
  put_u32(bytes, crc32(bytes.data(), bytes.size()));
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) return "nao consegui criar fixture ZTKZ";
  file.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  return file ? "" : "erro a escrever fixture ZTKZ";
}

void test_char_tokenizer() {
  Tokenizer tok;
  tok.kind = TokenizerKind::Char;
  tok.tokens = default_char_vocab();
  std::vector<uint32_t> ids;
  CHECK(encode_text(tok, "Olá!\n", ids).empty());
  CHECK(ids.size() == 5);
  std::string text;
  CHECK(decode_tokens(tok, ids, text).empty());
  CHECK(text == "Olá!\n");
  CHECK(!encode_text(tok, "🙂", ids).empty());
  CHECK(!decode_tokens(tok, {999}, text).empty());
}

void test_bpe_byte_fallback_and_merges() {
  Tokenizer tok = tiny_bpe_tokenizer();
  const std::string original = "abab Olá 🙂 <|user|>";
  std::vector<uint32_t> ids;
  CHECK(encode_text(tok, original, ids).empty());
  CHECK(!ids.empty());
  CHECK(ids[0] == 259);
  CHECK(ids[1] == 259);
  CHECK(ids.back() >= 3); // marker-looking user text is literal UTF-8 bytes
  std::string decoded;
  CHECK(decode_tokens(tok, ids, decoded, false).empty());
  CHECK(decoded == original);
  CHECK(!encode_text(tok, std::string("\xFF", 1), ids).empty());

  const std::vector<uint32_t> with_special = {0, 259, 1, 2};
  CHECK(decode_tokens(tok, with_special, decoded, true).empty());
  CHECK(decoded == "ab");
}

void test_bpe_zkt_reader_and_corruption() {
  const auto path = std::filesystem::temp_directory_path() / "zika_test_tokenizer.zkt";
  const Tokenizer expected = tiny_bpe_tokenizer();
  CHECK(write_bpe_fixture(path.string(), expected).empty());
  Tokenizer loaded;
  CHECK(read_tokenizer(path.string(), loaded).empty());
  CHECK(loaded.kind == TokenizerKind::BPE);
  CHECK(loaded.tokens == expected.tokens);
  CHECK(loaded.merges == expected.merges);

  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekg(-1, std::ios::end);
    char byte = 0;
    file.read(&byte, 1);
    byte = static_cast<char>(static_cast<unsigned char>(byte) ^ 0xFFu);
    file.seekp(-1, std::ios::end);
    file.write(&byte, 1);
  }
  Tokenizer corrupted;
  CHECK(!read_tokenizer(path.string(), corrupted).empty());
  std::filesystem::remove(path);
}

void test_bpe_model_format_and_generation() {
  Model model = tiny_bpe_model();
  std::string error;
  const auto bytes = serialize(model, error);
  CHECK(error.empty());
  Model restored;
  CHECK(deserialize(bytes.data(), bytes.size(), restored).empty());
  CHECK(restored.tok.kind == TokenizerKind::BPE);
  CHECK(restored.tok.tokens == model.tok.tokens);

  GenerationConfig cfg;
  cfg.max_new_tokens = 4;
  cfg.top_k = 1;
  cfg.top_p = 1.0f;
  cfg.temperature = 0.8f;
  std::mt19937 rng(42);
  std::vector<uint32_t> generated;
  CHECK(generate_tokens(restored, {0, 3, 1}, cfg, rng, generated).empty());
  for (uint32_t id : generated) CHECK(id >= 3 && id < restored.cfg.vocab_size);
  std::string generated_text;
  CHECK(decode_tokens(restored.tok, generated, generated_text, true).empty());
}

void test_sampling() {
  GenerationConfig cfg;
  cfg.temperature = 1.0f;
  cfg.top_k = 1;
  cfg.top_p = 1.0f;
  std::mt19937 rng(7);
  for (int i = 0; i < 20; ++i)
    CHECK(sample_next_token({0.0f, 10.0f, -2.0f}, cfg, rng) == 1);

  cfg.top_k = 0;
  cfg.top_p = 0.0f;
  bool threw = false;
  try { sample_next_token({1.0f, 2.0f}, cfg, rng); }
  catch (...) { threw = true; }
  CHECK(threw);
}

void test_char_generation() {
  const Model model = tiny_char_model();
  GenerationConfig cfg;
  cfg.max_new_tokens = 4;
  cfg.top_k = 0;
  cfg.top_p = 1.0f;
  cfg.temperature = 0.8f;
  std::mt19937 rng(42);
  std::vector<uint32_t> generated;
  CHECK(generate_tokens(model, {0, 3, 1}, cfg, rng, generated).empty());
  CHECK(generated.size() <= cfg.max_new_tokens);
  for (uint32_t id : generated) CHECK(id >= 3 && id < model.cfg.vocab_size);
}
} // namespace

int main() {
  test_char_tokenizer();
  test_bpe_byte_fallback_and_merges();
  test_bpe_zkt_reader_and_corruption();
  test_bpe_model_format_and_generation();
  test_sampling();
  test_char_generation();
  std::printf("ZIKA generation/tokenizers: %d verificacoes, %d falhas\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

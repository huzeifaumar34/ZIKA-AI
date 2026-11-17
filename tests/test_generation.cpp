#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "zika/generation.hpp"
#include "zika/transformer.hpp"

using namespace zika;

namespace {
int checks = 0;
int failures = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { ++failures; std::fprintf(stderr, "FALHOU %s:%d: %s\n", __FILE__, __LINE__, #condition); } } while (0)

Model tiny_model() {
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

void test_tokenizer() {
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

void test_generation() {
  const Model model = tiny_model();
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
  test_tokenizer();
  test_sampling();
  test_generation();
  std::printf("ZIKA generation: %d verificacoes, %d falhas\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "zika/format.hpp"
#include "zika/training.hpp"
#include "zika/transformer.hpp"

using namespace zika;

namespace {
int checks = 0;
int failures = 0;
#define CHECK(condition)                                                        \
  do {                                                                          \
    ++checks;                                                                   \
    if (!(condition)) {                                                         \
      ++failures;                                                               \
      std::fprintf(stderr, "FALHOU %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    }                                                                           \
  } while (0)

Model tiny_model() {
  Config cfg;
  cfg.vocab_size = 124;
  cfg.n_layers = 1;
  cfg.d_model = 8;
  cfg.n_heads = 2;
  cfg.d_ff = 16;
  cfg.ctx_len = 8;
  Tokenizer tok;
  tok.kind = TokenizerKind::Char;
  tok.tokens = default_char_vocab();
  Model model = make_empty_model(cfg, tok);
  initialize_weights(model, 77);
  return model;
}

std::vector<uint32_t> alternating(size_t count) {
  std::vector<uint32_t> ids;
  ids.reserve(count);
  for (size_t i = 0; i < count; ++i)
    ids.push_back(static_cast<uint32_t>(4 + (i % 2))); // " ! " ! ...
  return ids;
}

void test_overfit_and_checkpoint_resume() {
  const auto root = std::filesystem::temp_directory_path() / "zika_training_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::string checkpoint = (root / "nano.ckpt").string();
  const std::string model_path = (root / "nano.zika").string();

  const std::vector<uint32_t> train = alternating(128);
  const std::vector<uint32_t> validation = alternating(64);
  TrainingConfig cfg;
  cfg.steps = 80;
  cfg.batch_size = 2;
  cfg.sequence_length = 8;
  cfg.warmup_steps = 5;
  cfg.learning_rate = 0.01f;
  cfg.min_learning_rate_ratio = 0.25f;
  cfg.weight_decay = 0.0f;
  cfg.max_grad_norm = 1.0f;
  cfg.seed = 1234;
  cfg.log_every = 0;
  cfg.checkpoint_every = 20;

  Model model = tiny_model();
  TrainingResult result;
  const std::string error = train_model(model, train, validation, cfg,
                                        model_path, checkpoint, "", result);
  CHECK(error.empty());
  CHECK(result.initial_step == 0);
  CHECK(result.final_step == cfg.steps);
  CHECK(result.final_train_loss < result.initial_train_loss);
  CHECK(result.final_validation_loss < result.initial_train_loss);
  CHECK(std::filesystem::exists(checkpoint));
  CHECK(std::filesystem::exists(model_path));

  Model loaded;
  CHECK(read_zika(model_path, loaded).empty());
  CHECK(loaded.cfg.vocab_size == 124);

  // Retomar usa os pesos, momentos AdamW, passo e gerador aleatorio do checkpoint.
  Model resumed = tiny_model();
  cfg.steps = 90;
  TrainingResult resumed_result;
  const std::string resume_error = train_model(
      resumed, train, validation, cfg, "", checkpoint, checkpoint, resumed_result);
  CHECK(resume_error.empty());
  CHECK(resumed_result.initial_step == 80);
  CHECK(resumed_result.final_step == 90);

  // Checkpoints nao podem ser reutilizados com outro corpus.
  std::vector<uint32_t> other_validation = validation;
  other_validation[0] = 5;
  Model wrong_data = tiny_model();
  TrainingResult ignored;
  const std::string mismatch = train_model(
      wrong_data, train, other_validation, cfg, "", "", checkpoint, ignored);
  CHECK(mismatch.find("corpus nao corresponde") != std::string::npos);

  std::filesystem::remove_all(root);
}

void test_bad_token_file() {
  const auto path = std::filesystem::temp_directory_path() / "zika_bad_tokens.bin";
  {
    std::FILE *f = std::fopen(path.string().c_str(), "wb");
    const unsigned char bytes[] = {0, 0, 0, 0, 255};
    std::fwrite(bytes, 1, sizeof(bytes), f);
    std::fclose(f);
  }
  std::vector<uint32_t> tokens;
  CHECK(!load_token_ids(path.string(), 124, tokens).empty());
  std::filesystem::remove(path);
}
} // namespace

int main() {
  test_overfit_and_checkpoint_resume();
  test_bad_token_file();
  std::printf("ZIKA training: %d verificacoes, %d falhas\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

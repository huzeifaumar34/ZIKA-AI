#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <utility>
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

std::vector<uint32_t> encode_dialogues(
    const std::vector<std::pair<std::string, std::string>> &dialogues) {
  const auto vocab = default_char_vocab();
  std::vector<uint32_t> ids;
  for (const auto &dialogue : dialogues) {
    ids.push_back(0); // <|user|>
    for (char ch : dialogue.first) {
      const auto it = std::find(vocab.begin(), vocab.end(), std::string(1, ch));
      if (it == vocab.end()) throw std::runtime_error("caractere fora do vocabulario no teste");
      ids.push_back(static_cast<uint32_t>(std::distance(vocab.begin(), it)));
    }
    ids.push_back(1); // <|ia|>
    for (char ch : dialogue.second) {
      const auto it = std::find(vocab.begin(), vocab.end(), std::string(1, ch));
      if (it == vocab.end()) throw std::runtime_error("caractere fora do vocabulario no teste");
      ids.push_back(static_cast<uint32_t>(std::distance(vocab.begin(), it)));
    }
    ids.push_back(2); // <|fim|>
  }
  return ids;
}

std::vector<uint32_t> real_train_text() {
  return encode_dialogues({
      {"Como posso estudar?", "Estuda em blocos curtos e pratica todos os dias."},
      {"Como organizo o codigo?", "Separa as funcoes por responsabilidade e testa cada parte."},
      {"Para que serve o Git?", "Regista alteracoes e permite recuperar versoes do projeto."},
      {"Como aprender Python?", "Pratica com pequenos programas e consulta a documentacao."},
      {"Como proteger a conta?", "Usa uma palavra passe unica e ativa dois fatores."},
      {"Como corrigir um erro?", "Le a mensagem, reproduz o problema e testa uma mudanca."},
      {"O que e uma variavel?", "E um nome associado a um valor usado pelo programa."},
      {"O que e uma funcao?", "Agrupa instrucoes numa unidade que pode ser reutilizada."},
  });
}

std::vector<uint32_t> real_validation_text() {
  return encode_dialogues({
      {"Como melhorar a leitura?", "Escolhe um livro interessante e le um pouco diariamente."},
      {"Como guardar ficheiros?", "Organiza pastas por projeto e mantem copias de seguranca."},
      {"Como aprender a programar?", "Resolve problemas pequenos e constroi projetos simples."},
  });
}

void test_real_text_training_and_sampled_evaluation() {
  const auto train = real_train_text();
  const auto validation = real_validation_text();
  Model model = tiny_model();

  const float initial_validation = evaluate_loss(model, validation, 8, 0);
  const float sampled_validation = evaluate_loss(model, validation, 8, 32);
  CHECK(std::isfinite(initial_validation));
  CHECK(std::isfinite(sampled_validation));
  CHECK(std::abs(initial_validation - sampled_validation) > 1.0e-7f);

  TrainingConfig cfg;
  cfg.steps = 120;
  cfg.batch_size = 2;
  cfg.sequence_length = 8;
  cfg.warmup_steps = 10;
  cfg.learning_rate = 0.005f;
  cfg.min_learning_rate_ratio = 0.25f;
  cfg.weight_decay = 0.0f;
  cfg.max_grad_norm = 1.0f;
  cfg.seed = 1234;
  cfg.log_every = 0;
  cfg.checkpoint_every = 0;
  cfg.eval_every = 40;
  cfg.eval_tokens = 32;
  cfg.full_eval_every = 0;

  TrainingResult result;
  const std::string error = train_model(model, train, validation, cfg, "", "", "", result);
  CHECK(error.empty());
  CHECK(result.final_step == cfg.steps);
  CHECK(result.final_train_loss < result.initial_train_loss);
  CHECK(std::isfinite(result.initial_validation_loss));
  CHECK(std::isfinite(result.final_validation_loss));
  CHECK(result.sampled_evaluations >= 2);
  CHECK(result.full_evaluations == 1);
  CHECK(result.final_validation_loss < initial_validation);
}

void test_checkpoint_extension_is_smooth_and_data_bound() {
  const auto root = std::filesystem::temp_directory_path() / "zika_training_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::string checkpoint = (root / "nano.ckpt").string();

  const auto train = real_train_text();
  const auto validation = real_validation_text();
  TrainingConfig cfg;
  cfg.steps = 80;
  cfg.batch_size = 2;
  cfg.sequence_length = 8;
  cfg.warmup_steps = 8;
  cfg.learning_rate = 0.005f;
  cfg.min_learning_rate_ratio = 0.25f;
  cfg.weight_decay = 0.0f;
  cfg.max_grad_norm = 1.0f;
  cfg.seed = 1234;
  cfg.log_every = 0;
  cfg.checkpoint_every = 20;
  cfg.eval_every = 0;
  cfg.full_eval_every = 0;

  Model model = tiny_model();
  TrainingResult first;
  CHECK(train_model(model, train, validation, cfg, "", checkpoint, "", first).empty());
  CHECK(first.final_step == 80);
  CHECK(std::filesystem::exists(checkpoint));

  Model resumed = tiny_model();
  cfg.steps = 120; // prolonga o treino ja terminado: deve ramp-up sem salto abrupto
  cfg.resume_warmup_steps = 10;
  TrainingResult continued;
  CHECK(train_model(resumed, train, validation, cfg, "", checkpoint, checkpoint,
                    continued).empty());
  CHECK(continued.initial_step == 80);
  CHECK(continued.final_step == 120);
  CHECK(continued.initial_learning_rate > 0.0f);
  CHECK(continued.first_learning_rate >= continued.initial_learning_rate);
  CHECK(continued.first_learning_rate <= continued.initial_learning_rate * 2.0f);

  // Um checkpoint nao pode ser reaproveitado com outro corpus.
  auto other_validation = validation;
  other_validation[0] = 5;
  Model wrong_data = tiny_model();
  TrainingResult ignored;
  const std::string mismatch = train_model(
      wrong_data, train, other_validation, cfg, "", "", checkpoint, ignored);
  CHECK(mismatch.find("corpus nao corresponde") != std::string::npos);

  // Nem se deve alterar silenciosamente a configuracao que afeta os gradientes.
  Model wrong_config = tiny_model();
  TrainingConfig incompatible = cfg;
  incompatible.batch_size = 1;
  const std::string config_error = train_model(
      wrong_config, train, validation, incompatible, "", "", checkpoint, ignored);
  CHECK(config_error.find("configuracao de treino difere") != std::string::npos);

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
  test_real_text_training_and_sampled_evaluation();
  test_checkpoint_extension_is_smooth_and_data_bound();
  test_bad_token_file();
  std::printf("ZIKA training: %d verificacoes, %d falhas\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

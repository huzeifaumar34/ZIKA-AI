// Treino causal, avaliacao deterministica e checkpoints retomaveis do ZIKA-AI.
#pragma once

#include "zika/format.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zika {

struct TrainingConfig {
  uint64_t steps = 500;                 // passo total pretendido, nao passos adicionais
  uint32_t batch_size = 1;
  uint32_t sequence_length = 128;
  uint64_t warmup_steps = 50;
  float learning_rate = 3.0e-4f;
  float min_learning_rate_ratio = 0.1f;
  float weight_decay = 0.01f;
  float max_grad_norm = 1.0f;
  uint32_t seed = 1337;
  uint32_t log_every = 25;
  uint64_t checkpoint_every = 100;
  uint32_t eval_every = 100;             // 0 desativa avaliacao amostrada periodica
  uint32_t eval_tokens = 2048;           // maximo de tokens por avaliacao amostrada
  uint64_t full_eval_every = 500;        // 0 desativa avaliacao completa periodica
  uint32_t resume_warmup_steps = 50;     // rampa suave ao prolongar um treino ja concluido
  uint64_t legacy_resume_total_steps = 0; // obrigatorio apenas para checkpoints ZCKP v1
};

struct TrainingResult {
  uint64_t initial_step = 0;
  uint64_t final_step = 0;
  float initial_train_loss = 0.0f;
  float initial_validation_loss = 0.0f;
  float final_train_loss = 0.0f;
  float final_validation_loss = 0.0f;
  float initial_learning_rate = 0.0f;
  float first_learning_rate = 0.0f;
  float final_learning_rate = 0.0f;
  uint64_t sampled_evaluations = 0;
  uint64_t full_evaluations = 0;
  double elapsed_seconds = 0.0;
};

std::string load_token_ids(const std::string &path, uint32_t vocab_size,
                           std::vector<uint32_t> &out);

// max_tokens == 0 avalia todo o corpus. Caso contrario, avalia janelas
// deterministicas distribuidas ao longo do corpus, sem favorecer o prefixo.
float evaluate_loss(const Model &model, const std::vector<uint32_t> &tokens,
                    uint32_t sequence_length, uint64_t max_tokens = 0);

std::string train_model(Model &model,
                        const std::vector<uint32_t> &train_tokens,
                        const std::vector<uint32_t> &validation_tokens,
                        const TrainingConfig &config,
                        const std::string &model_path,
                        const std::string &checkpoint_path,
                        const std::string &resume_path,
                        TrainingResult &result);

} // namespace zika

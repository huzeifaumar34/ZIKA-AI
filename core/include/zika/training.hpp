// Ciclo de treino causal, scheduler e checkpoints retomaveis do ZIKA-AI.
#pragma once

#include "zika/format.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zika {

struct TrainingConfig {
  uint64_t steps = 500;
  uint32_t batch_size = 1;
  uint32_t sequence_length = 128;
  uint64_t warmup_steps = 50;
  float learning_rate = 3.0e-4f;
  float min_learning_rate_ratio = 0.1f;
  float weight_decay = 0.01f;
  float max_grad_norm = 1.0f;
  uint32_t seed = 1337;
  uint32_t log_every = 25; // 0 desativa o log periodico
  uint64_t checkpoint_every = 100; // 0 grava apenas no fim
};

struct TrainingResult {
  uint64_t initial_step = 0;
  uint64_t final_step = 0;
  float initial_train_loss = 0.0f;
  float final_train_loss = 0.0f;
  float final_validation_loss = 0.0f;
  double elapsed_seconds = 0.0;
};

// Lê IDs uint32 little-endian e rejeita ficheiros vazios, truncados ou IDs inválidos.
std::string load_token_ids(const std::string &path, uint32_t vocab_size,
                           std::vector<uint32_t> &out);

// Avalia cross-entropy causal em janelas sequenciais determinísticas.
float evaluate_loss(const Model &model, const std::vector<uint32_t> &tokens,
                    uint32_t sequence_length);

// Executa o treino. steps representa o número total de passos pretendido.
// resume_path pode apontar para um checkpoint; checkpoint_path e model_path
// são escritos de forma atómica quando não estão vazios.
std::string train_model(Model &model,
                        const std::vector<uint32_t> &train_tokens,
                        const std::vector<uint32_t> &validation_tokens,
                        const TrainingConfig &config,
                        const std::string &model_path,
                        const std::string &checkpoint_path,
                        const std::string &resume_path,
                        TrainingResult &result);

} // namespace zika

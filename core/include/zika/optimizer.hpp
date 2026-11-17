// Otimizador AdamW com clipping global de gradientes.
#pragma once

#include "zika/format.hpp"

#include <cstdint>
#include <vector>

namespace zika {

struct AdamWConfig {
  float learning_rate = 3.0e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1.0e-8f;
  float weight_decay = 0.01f;
  float max_grad_norm = 1.0f; // <= 0 desativa clipping
};

struct OptimizerStats {
  double grad_norm = 0.0;
  float clip_scale = 1.0f;
  uint64_t step = 0;
};

class AdamW {
public:
  explicit AdamW(const Model &model, AdamWConfig config = {});
  OptimizerStats step(Model &model, const Model &gradients);
  uint64_t step_count() const noexcept { return step_; }

private:
  AdamWConfig config_;
  uint64_t step_ = 0;
  std::vector<std::vector<float>> first_moment_;
  std::vector<std::vector<float>> second_moment_;
};

} // namespace zika

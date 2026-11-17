#include "zika/optimizer.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace zika {
namespace {
bool no_weight_decay(const std::string &name) {
  return name.find(".b") != std::string::npos ||
         (!name.empty() && name.back() == 'g' &&
          name.size() >= 2 && name.compare(name.size() - 2, 2, ".g") == 0);
}
void check_compatible(const Model &model, const Model &grads) {
  if (model.tensors.size() != grads.tensors.size())
    throw std::invalid_argument("AdamW: quantidade de tensores incompativel");
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    const Tensor &p = model.tensors[i], &g = grads.tensors[i];
    if (p.name != g.name || p.shape != g.shape || p.data.size() != g.data.size())
      throw std::invalid_argument("AdamW: tensor incompativel: " + p.name);
  }
}
}

AdamW::AdamW(const Model &model, AdamWConfig config) : config_(config) {
  if (!(config.learning_rate > 0.0f) ||
      !(config.beta1 >= 0.0f && config.beta1 < 1.0f) ||
      !(config.beta2 >= 0.0f && config.beta2 < 1.0f) ||
      !(config.epsilon > 0.0f) || config.weight_decay < 0.0f)
    throw std::invalid_argument("AdamW: hiperparametros invalidos");
  first_moment_.reserve(model.tensors.size());
  second_moment_.reserve(model.tensors.size());
  for (const Tensor &t : model.tensors) {
    first_moment_.emplace_back(t.data.size(), 0.0f);
    second_moment_.emplace_back(t.data.size(), 0.0f);
  }
}

OptimizerStats AdamW::step(Model &model, const Model &gradients) {
  check_compatible(model, gradients);
  if (model.tensors.size() != first_moment_.size())
    throw std::invalid_argument("AdamW: modelo diferente do usado no construtor");
  double norm2 = 0.0;
  for (const Tensor &g : gradients.tensors)
    for (float v : g.data) {
      if (!std::isfinite(v))
        throw std::runtime_error("AdamW: gradiente NaN ou infinito");
      norm2 += static_cast<double>(v) * v;
    }
  const double norm = std::sqrt(norm2);
  float scale = 1.0f;
  if (config_.max_grad_norm > 0.0f && norm > config_.max_grad_norm)
    scale = static_cast<float>(config_.max_grad_norm / (norm + 1.0e-12));
  ++step_;
  const double b1_correction = 1.0 - std::pow(config_.beta1, static_cast<double>(step_));
  const double b2_correction = 1.0 - std::pow(config_.beta2, static_cast<double>(step_));

  for (size_t ti = 0; ti < model.tensors.size(); ++ti) {
    Tensor &p = model.tensors[ti];
    const Tensor &g = gradients.tensors[ti];
    auto &m = first_moment_[ti];
    auto &v = second_moment_[ti];
    const bool decay = !no_weight_decay(p.name);
    for (size_t i = 0; i < p.data.size(); ++i) {
      const float grad = g.data[i] * scale;
      m[i] = config_.beta1 * m[i] + (1.0f - config_.beta1) * grad;
      v[i] = config_.beta2 * v[i] + (1.0f - config_.beta2) * grad * grad;
      const double mh = m[i] / b1_correction;
      const double vh = v[i] / b2_correction;
      const float update = static_cast<float>(mh / (std::sqrt(vh) + config_.epsilon));
      if (decay) p.data[i] *= (1.0f - config_.learning_rate * config_.weight_decay);
      p.data[i] -= config_.learning_rate * update;
    }
  }
  return {norm, scale, step_};
}

} // namespace zika

#include "zika/optimizer.hpp"

#include <cmath>
#include <cstring>
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
struct Writer {
  std::vector<uint8_t> bytes;
  void u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void f32(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    u32(bits);
  }
};
struct Reader {
  const uint8_t *p;
  const uint8_t *end;
  size_t left() const { return static_cast<size_t>(end - p); }
  bool u32(uint32_t &v) {
    if (left() < 4) return false;
    v = static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return true;
  }
  bool u64(uint64_t &v) {
    if (left() < 8) return false;
    v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    p += 8;
    return true;
  }
  bool f32(float &f) {
    uint32_t bits = 0;
    if (!u32(bits)) return false;
    std::memcpy(&f, &bits, sizeof(f));
    return true;
  }
};
} // namespace

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

void AdamW::set_learning_rate(float learning_rate) {
  if (!(learning_rate > 0.0f) || !std::isfinite(learning_rate))
    throw std::invalid_argument("AdamW: learning_rate invalido");
  config_.learning_rate = learning_rate;
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

std::vector<uint8_t> AdamW::save_state() const {
  Writer w;
  w.bytes.insert(w.bytes.end(), {'Z', 'O', 'P', 'T'});
  w.u32(1);
  w.u64(step_);
  w.u32(static_cast<uint32_t>(first_moment_.size()));
  for (size_t i = 0; i < first_moment_.size(); ++i) {
    w.u64(static_cast<uint64_t>(first_moment_[i].size()));
    for (float v : first_moment_[i]) w.f32(v);
    for (float v : second_moment_[i]) w.f32(v);
  }
  w.u32(crc32(w.bytes.data(), w.bytes.size()));
  return std::move(w.bytes);
}

std::string AdamW::load_state(const uint8_t *data, size_t size, const Model &model) {
  if (data == nullptr || size < 4 + 4 + 8 + 4 + 4)
    return "AdamW: estado truncado";
  if (std::memcmp(data, "ZOPT", 4) != 0)
    return "AdamW: magic de estado invalida";
  const uint8_t *tail = data + size - 4;
  const uint32_t stored =
      static_cast<uint32_t>(tail[0]) |
      (static_cast<uint32_t>(tail[1]) << 8) |
      (static_cast<uint32_t>(tail[2]) << 16) |
      (static_cast<uint32_t>(tail[3]) << 24);
  if (crc32(data, size - 4) != stored)
    return "AdamW: checksum do estado invalido";

  Reader r{data + 4, tail};
  uint32_t version = 0, count = 0;
  uint64_t step = 0;
  if (!r.u32(version) || version != 1 || !r.u64(step) || !r.u32(count))
    return "AdamW: cabecalho de estado invalido";
  if (count != model.tensors.size() || count != first_moment_.size())
    return "AdamW: quantidade de tensores do estado incompativel";

  std::vector<std::vector<float>> first, second;
  first.reserve(count);
  second.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t n = 0;
    if (!r.u64(n) || n != model.tensors[i].data.size() ||
        n > static_cast<uint64_t>(r.left() / 8))
      return "AdamW: dimensao do estado incompativel";
    first.emplace_back(static_cast<size_t>(n));
    second.emplace_back(static_cast<size_t>(n));
    for (float &v : first.back())
      if (!r.f32(v) || !std::isfinite(v)) return "AdamW: primeiro momento invalido";
    for (float &v : second.back())
      if (!r.f32(v) || !std::isfinite(v) || v < 0.0f)
        return "AdamW: segundo momento invalido";
  }
  if (r.left() != 0)
    return "AdamW: dados inesperados no estado";
  first_moment_ = std::move(first);
  second_moment_ = std::move(second);
  step_ = step;
  return "";
}

} // namespace zika

#include "zika/transformer.hpp"

#include "zika/ops.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace zika {
namespace {

using ops::matrix;
constexpr float kLayerNormEpsilon = 1.0e-5f;

size_t param_index(const Model &model, const std::string &name) {
  for (size_t i = 0; i < model.tensors.size(); ++i)
    if (model.tensors[i].name == name) return i;
  throw std::invalid_argument("parametro ausente no modelo: " + name);
}
const Tensor &param(const Model &model, const char *name) {
  return model.tensors[param_index(model, std::string(name))];
}
const Tensor &param(const Model &model, const std::string &name) {
  return model.tensors[param_index(model, name)];
}
Tensor &grad_param(Model &grads, const std::string &name) {
  return grads.tensors[param_index(grads, name)];
}

Tensor add_bias(const Tensor &x, const Tensor &bias) {
  if (x.shape.size() != 2 || bias.shape.size() != 1 ||
      x.shape[1] != bias.shape[0])
    throw std::invalid_argument("linear: dimensoes do bias invalidas");
  Tensor y = x;
  const uint32_t rows = x.shape[0], cols = x.shape[1];
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < cols; ++c)
      y.data[static_cast<size_t>(r) * cols + c] += bias.data[c];
  return y;
}

Tensor linear(const Tensor &x, const Tensor &weight, const Tensor &bias) {
  return add_bias(ops::matmul(x, weight), bias);
}

void linear_backward(const Tensor &x, const Tensor &weight, const Tensor &dy,
                     Tensor &dx, Tensor &dweight, Tensor &dbias) {
  const uint32_t rows = x.shape[0], in = x.shape[1], out = weight.shape[1];
  if (weight.shape[0] != in || dy.shape != std::vector<uint32_t>{rows, out})
    throw std::invalid_argument("linear_backward: dimensoes incompativeis");
  Tensor wt = ops::transpose(weight);
  dx = ops::matmul(dy, wt);
  Tensor xt = ops::transpose(x);
  Tensor dw = ops::matmul(xt, dy);
  if (dweight.shape != weight.shape) dweight = matrix(in, out);
  for (size_t i = 0; i < dw.data.size(); ++i) dweight.data[i] += dw.data[i];
  if (dbias.shape != std::vector<uint32_t>{out}) {
    dbias.shape = {out}; dbias.data.assign(out, 0.0f);
  }
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < out; ++c)
      dbias.data[c] += dy.data[static_cast<size_t>(r) * out + c];
}

struct LayerCache {
  Tensor input;
  Tensor ln1;
  Tensor qkv;
  Tensor probabilities; // [heads, T, T], causal
  Tensor context;       // [T, D]
  Tensor residual1;
  Tensor ln2;
  Tensor mlp_pre;
  Tensor mlp_gelu;
};

struct ForwardCache {
  uint32_t t = 0, d = 0, h = 0, dh = 0, f = 0, vocab = 0;
  std::vector<uint32_t> ids;
  Tensor embeddings;
  std::vector<LayerCache> layers;
  Tensor final_norm_input;
  Tensor final_norm;
  Tensor logits;
};

ForwardCache run_forward(const Model &model, const std::vector<uint32_t> &ids) {
  const Config &cfg = model.cfg;
  const std::string config_error = validate_config(cfg);
  if (!config_error.empty()) throw std::invalid_argument(config_error);
  if (ids.empty() || ids.size() > cfg.ctx_len)
    throw std::invalid_argument("sequencia vazia ou maior que ctx_len");
  if (model.tensors.size() != tensor_specs(cfg).size())
    throw std::invalid_argument("modelo nao contem todos os tensores");

  ForwardCache c;
  c.t = static_cast<uint32_t>(ids.size());
  c.d = cfg.d_model; c.h = cfg.n_heads; c.dh = cfg.d_model / cfg.n_heads;
  c.f = cfg.d_ff; c.vocab = cfg.vocab_size; c.ids = ids;
  c.embeddings = matrix(c.t, c.d);
  const Tensor &emb = param(model, "tok_emb");
  const Tensor &pos = param(model, "pos_emb");
  for (uint32_t t = 0; t < c.t; ++t) {
    if (ids[t] >= cfg.vocab_size)
      throw std::invalid_argument("token de entrada fora do vocabulario");
    for (uint32_t j = 0; j < c.d; ++j)
      c.embeddings.data[static_cast<size_t>(t) * c.d + j] =
          emb.data[static_cast<size_t>(ids[t]) * c.d + j] +
          pos.data[static_cast<size_t>(t) * c.d + j];
  }

  Tensor x = c.embeddings;
  c.layers.reserve(cfg.n_layers);
  const float scale = 1.0f / std::sqrt(static_cast<float>(c.dh));
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const std::string p = "layer" + std::to_string(l) + ".";
    LayerCache lc;
    lc.input = x;
    lc.ln1 = ops::layer_norm(x, param(model, p + "ln1.g"),
                             param(model, p + "ln1.b"), kLayerNormEpsilon);
    lc.qkv = linear(lc.ln1, param(model, p + "attn.wqkv"),
                    param(model, p + "attn.bqkv"));
    lc.probabilities = matrix(c.h * c.t, c.t);
    lc.context = matrix(c.t, c.d);

    // Atencao multi-cabeca causal. As colunas futuras recebem -1e30 antes do softmax.
    for (uint32_t head = 0; head < c.h; ++head) {
      Tensor scores = matrix(c.t, c.t, -1.0e30f);
      for (uint32_t i = 0; i < c.t; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
          float dot = 0.0f;
          for (uint32_t k = 0; k < c.dh; ++k) {
            const size_t qi = static_cast<size_t>(i) * 3 * c.d + head * c.dh + k;
            const size_t kj = static_cast<size_t>(j) * 3 * c.d + c.d + head * c.dh + k;
            dot += lc.qkv.data[qi] * lc.qkv.data[kj];
          }
          scores.data[static_cast<size_t>(i) * c.t + j] = dot * scale;
        }
      }
      Tensor probs = ops::softmax_rows(scores);
      for (uint32_t i = 0; i < c.t; ++i)
        for (uint32_t j = 0; j < c.t; ++j)
          lc.probabilities.data[(static_cast<size_t>(head) * c.t + i) * c.t + j] =
              probs.data[static_cast<size_t>(i) * c.t + j];

      for (uint32_t i = 0; i < c.t; ++i) {
        for (uint32_t k = 0; k < c.dh; ++k) {
          float value = 0.0f;
          for (uint32_t j = 0; j <= i; ++j) {
            const size_t vi = static_cast<size_t>(j) * 3 * c.d +
                              2 * c.d + head * c.dh + k;
            value += probs.data[static_cast<size_t>(i) * c.t + j] *
                     lc.qkv.data[vi];
          }
          lc.context.data[static_cast<size_t>(i) * c.d + head * c.dh + k] = value;
        }
      }
    }
    Tensor attn = linear(lc.context, param(model, p + "attn.wo"),
                         param(model, p + "attn.bo"));
    lc.residual1 = x;
    for (size_t i = 0; i < x.data.size(); ++i)
      lc.residual1.data[i] += attn.data[i];

    lc.ln2 = ops::layer_norm(lc.residual1, param(model, p + "ln2.g"),
                             param(model, p + "ln2.b"), kLayerNormEpsilon);
    lc.mlp_pre = linear(lc.ln2, param(model, p + "mlp.w1"),
                        param(model, p + "mlp.b1"));
    lc.mlp_gelu = ops::gelu(lc.mlp_pre);
    Tensor mlp_out = linear(lc.mlp_gelu, param(model, p + "mlp.w2"),
                            param(model, p + "mlp.b2"));
    x = lc.residual1;
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] += mlp_out.data[i];
    c.layers.push_back(std::move(lc));
  }

  c.final_norm_input = x;
  c.final_norm = ops::layer_norm(x, param(model, "lnf.g"),
                                 param(model, "lnf.b"), kLayerNormEpsilon);
  c.logits = ops::matmul(c.final_norm, ops::transpose(emb));
  return c;
}

Model zero_gradient_model(const Model &model) {
  Model g;
  g.cfg = model.cfg;
  g.dtype = model.dtype;
  g.tok = model.tok;
  g.tensors.reserve(model.tensors.size());
  for (const Tensor &p : model.tensors) {
    Tensor t;
    t.name = p.name;
    t.shape = p.shape;
    t.data.assign(p.data.size(), 0.0f);
    g.tensors.push_back(std::move(t));
  }
  return g;
}

} // namespace

void initialize_weights(Model &model, uint32_t seed) {
  const std::string err = validate_config(model.cfg);
  if (!err.empty()) throw std::invalid_argument(err);
  if (model.tensors.size() != tensor_specs(model.cfg).size())
    throw std::invalid_argument("modelo incompleto para inicializacao");
  std::mt19937 rng(seed);
  std::normal_distribution<float> embedding_dist(0.0f, 0.02f);
  for (Tensor &t : model.tensors) {
    const bool norm_gain = t.name.size() >= 2 &&
                           t.name.compare(t.name.size() - 2, 2, ".g") == 0;
    const bool bias = t.name.find(".b") != std::string::npos;
    if (norm_gain) {
      std::fill(t.data.begin(), t.data.end(), 1.0f);
    } else if (bias) {
      std::fill(t.data.begin(), t.data.end(), 0.0f);
    } else if (t.name == "tok_emb" || t.name == "pos_emb") {
      for (float &v : t.data) v = embedding_dist(rng);
    } else if (t.shape.size() == 2) {
      const float fan_in = static_cast<float>(t.shape[0]);
      const float fan_out = static_cast<float>(t.shape[1]);
      const float bound = std::sqrt(6.0f / (fan_in + fan_out));
      std::uniform_real_distribution<float> dist(-bound, bound);
      for (float &v : t.data) v = dist(rng);
    } else {
      std::fill(t.data.begin(), t.data.end(), 0.0f);
    }
  }
}

std::vector<float> forward_logits(const Model &model,
                                  const std::vector<uint32_t> &input_ids) {
  return run_forward(model, input_ids).logits.data;
}

float loss_and_backward(const Model &model,
                        const std::vector<uint32_t> &input_ids,
                        const std::vector<uint32_t> &targets,
                        Model &gradients) {
  ForwardCache c = run_forward(model, input_ids);
  if (targets.size() != c.t)
    throw std::invalid_argument("targets tem de ter o mesmo tamanho que input_ids");
  for (uint32_t id : targets)
    if (id >= c.vocab) throw std::invalid_argument("target fora do vocabulario");

  gradients = zero_gradient_model(model);
  Tensor dlogits = matrix(c.t, c.vocab);
  double loss = 0.0;
  for (uint32_t t = 0; t < c.t; ++t) {
    const size_t base = static_cast<size_t>(t) * c.vocab;
    float maximum = c.logits.data[base];
    for (uint32_t v = 1; v < c.vocab; ++v)
      maximum = std::max(maximum, c.logits.data[base + v]);
    double total = 0.0;
    for (uint32_t v = 0; v < c.vocab; ++v) {
      const float e = std::exp(c.logits.data[base + v] - maximum);
      dlogits.data[base + v] = e;
      total += e;
    }
    loss += static_cast<double>(maximum) + std::log(total) -
            c.logits.data[base + targets[t]];
    for (uint32_t v = 0; v < c.vocab; ++v)
      dlogits.data[base + v] /= static_cast<float>(total);
    dlogits.data[base + targets[t]] -= 1.0f;
  }
  const float inv_t = 1.0f / static_cast<float>(c.t);
  for (float &v : dlogits.data) v *= inv_t;
  const float mean_loss = static_cast<float>(loss / c.t);

  // Weight tying: E serve simultaneamente a entrada e à projeccao de saída.
  Tensor &demb = grad_param(gradients, "tok_emb");
  Tensor d_final = ops::matmul(dlogits, param(model, "tok_emb"));
  Tensor d_emb_out = ops::matmul(ops::transpose(dlogits), c.final_norm);
  for (size_t i = 0; i < demb.data.size(); ++i)
    demb.data[i] += d_emb_out.data[i];

  Tensor dx;
  ops::layer_norm_backward(c.final_norm_input, param(model, "lnf.g"),
                           d_final, dx, grad_param(gradients, "lnf.g"),
                           grad_param(gradients, "lnf.b"), kLayerNormEpsilon);

  for (int li = static_cast<int>(c.layers.size()) - 1; li >= 0; --li) {
    const uint32_t l = static_cast<uint32_t>(li);
    const std::string p = "layer" + std::to_string(l) + ".";
    const LayerCache &lc = c.layers[l];

    // Residual + MLP, na ordem inversa do forward.
    Tensor d_residual1 = dx;
    Tensor d_mlp2 = dx;
    Tensor d_gelu, d_mlp2_w, d_mlp2_b;
    linear_backward(lc.mlp_gelu, param(model, p + "mlp.w2"), d_mlp2,
                    d_gelu, grad_param(gradients, p + "mlp.w2"),
                    grad_param(gradients, p + "mlp.b2"));
    Tensor d_mlp_pre = ops::gelu_backward(lc.mlp_pre, d_gelu);
    Tensor d_ln2, d_mlp1_w, d_mlp1_b;
    linear_backward(lc.ln2, param(model, p + "mlp.w1"), d_mlp_pre,
                    d_ln2, grad_param(gradients, p + "mlp.w1"),
                    grad_param(gradients, p + "mlp.b1"));
    Tensor d_residual1_ln;
    ops::layer_norm_backward(lc.residual1, param(model, p + "ln2.g"),
                             d_ln2, d_residual1_ln,
                             grad_param(gradients, p + "ln2.g"),
                             grad_param(gradients, p + "ln2.b"),
                             kLayerNormEpsilon);
    for (size_t i = 0; i < d_residual1.data.size(); ++i)
      d_residual1.data[i] += d_residual1_ln.data[i];

    // Residual + atencao.
    Tensor d_context, d_attn_w, d_attn_b;
    linear_backward(lc.context, param(model, p + "attn.wo"), d_residual1,
                    d_context, grad_param(gradients, p + "attn.wo"),
                    grad_param(gradients, p + "attn.bo"));

    Tensor dqkv = matrix(c.t, 3 * c.d);
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.dh));
    for (uint32_t head = 0; head < c.h; ++head) {
      Tensor probs = matrix(c.t, c.t);
      Tensor dprobs = matrix(c.t, c.t);
      for (uint32_t i = 0; i < c.t; ++i)
        for (uint32_t j = 0; j < c.t; ++j)
          probs.data[static_cast<size_t>(i) * c.t + j] =
              lc.probabilities.data[(static_cast<size_t>(head) * c.t + i) * c.t + j];

      Tensor dv = matrix(c.t, c.dh);
      for (uint32_t i = 0; i < c.t; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
          const float probability = probs.data[static_cast<size_t>(i) * c.t + j];
          float dot = 0.0f;
          for (uint32_t k = 0; k < c.dh; ++k) {
            const float dc = d_context.data[static_cast<size_t>(i) * c.d +
                                             head * c.dh + k];
            const float value = lc.qkv.data[static_cast<size_t>(j) * 3 * c.d +
                                              2 * c.d + head * c.dh + k];
            dot += dc * value;
            dv.data[static_cast<size_t>(j) * c.dh + k] += probability * dc;
          }
          dprobs.data[static_cast<size_t>(i) * c.t + j] = dot;
        }
      }
      Tensor dscores = ops::softmax_backward(probs, dprobs);
      for (uint32_t i = 0; i < c.t; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
          const float ds = dscores.data[static_cast<size_t>(i) * c.t + j] * scale;
          for (uint32_t k = 0; k < c.dh; ++k) {
            const size_t qi = static_cast<size_t>(i) * 3 * c.d +
                              head * c.dh + k;
            const size_t kj = static_cast<size_t>(j) * 3 * c.d +
                              c.d + head * c.dh + k;
            const float q = lc.qkv.data[qi];
            const float key = lc.qkv.data[kj];
            dqkv.data[qi] += ds * key;
            dqkv.data[kj] += ds * q;
          }
        }
      }
      for (uint32_t j = 0; j < c.t; ++j)
        for (uint32_t k = 0; k < c.dh; ++k)
          dqkv.data[static_cast<size_t>(j) * 3 * c.d +
                    2 * c.d + head * c.dh + k] =
              dv.data[static_cast<size_t>(j) * c.dh + k];
    }

    Tensor d_ln1, d_qkv_w, d_qkv_b;
    linear_backward(lc.ln1, param(model, p + "attn.wqkv"), dqkv,
                    d_ln1, grad_param(gradients, p + "attn.wqkv"),
                    grad_param(gradients, p + "attn.bqkv"));
    Tensor d_input_ln;
    ops::layer_norm_backward(lc.input, param(model, p + "ln1.g"),
                             d_ln1, d_input_ln,
                             grad_param(gradients, p + "ln1.g"),
                             grad_param(gradients, p + "ln1.b"),
                             kLayerNormEpsilon);
    dx = d_residual1;
    for (size_t i = 0; i < dx.data.size(); ++i)
      dx.data[i] += d_input_ln.data[i];
  }

  // Gradientes dos embeddings de entrada (além da utilização tied na saída).
  Tensor &dpos = grad_param(gradients, "pos_emb");
  for (uint32_t t = 0; t < c.t; ++t) {
    for (uint32_t j = 0; j < c.d; ++j) {
      const float g = dx.data[static_cast<size_t>(t) * c.d + j];
      demb.data[static_cast<size_t>(c.ids[t]) * c.d + j] += g;
      dpos.data[static_cast<size_t>(t) * c.d + j] += g;
    }
  }
  return mean_loss;
}

} // namespace zika

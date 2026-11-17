#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "zika/ops.hpp"
#include "zika/optimizer.hpp"
#include "zika/transformer.hpp"

using namespace zika;
using namespace zika::ops;

namespace {
int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::fprintf(stderr, "FALHOU %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

float objective(const Tensor &y, const Tensor &upstream) {
  double s = 0.0;
  for (size_t i = 0; i < y.data.size(); ++i) s += y.data[i] * upstream.data[i];
  return static_cast<float>(s);
}
bool close_grad(float a, float b, float atol = 2.0e-3f, float rtol = 2.0e-2f) {
  return std::fabs(a - b) <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
}
float finite_difference(Tensor &x, size_t index, const std::function<float()> &fn) {
  const float original = x.data[index], eps = 1.0e-3f;
  x.data[index] = original + eps; const float plus = fn();
  x.data[index] = original - eps; const float minus = fn();
  x.data[index] = original;
  return (plus - minus) / (2.0f * eps);
}

void test_matmul_gradient() {
  Tensor a = matrix(2, 3), b = matrix(3, 2), upstream = matrix(2, 2);
  for (size_t i = 0; i < a.data.size(); ++i) a.data[i] = 0.13f * (i + 1) - 0.2f;
  for (size_t i = 0; i < b.data.size(); ++i) b.data[i] = -0.07f * (i + 1) + 0.3f;
  for (size_t i = 0; i < upstream.data.size(); ++i) upstream.data[i] = 0.17f * (i + 1);
  Tensor da = matrix(2, 3), db = matrix(3, 2);
  matmul_backward(a, b, upstream, da, db);
  for (size_t i = 0; i < a.data.size(); ++i) {
    const float numeric = finite_difference(a, i, [&] { return objective(matmul(a, b), upstream); });
    CHECK(close_grad(da.data[i], numeric));
  }
  for (size_t i = 0; i < b.data.size(); ++i) {
    const float numeric = finite_difference(b, i, [&] { return objective(matmul(a, b), upstream); });
    CHECK(close_grad(db.data[i], numeric));
  }
}

void test_layer_norm_gradient() {
  Tensor x = matrix(2, 4), gamma; gamma.shape = {4}; gamma.data = {0.8f, 1.1f, -0.6f, 0.4f};
  Tensor beta; beta.shape = {4}; beta.data = {0.1f, -0.2f, 0.05f, 0.3f};
  Tensor upstream = matrix(2, 4);
  for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = 0.21f * (i + 1) - 0.5f;
  for (size_t i = 0; i < upstream.data.size(); ++i) upstream.data[i] = 0.11f * (i + 1) - 0.2f;
  Tensor dx = matrix(2, 4), dg; dg.shape = {4}; dg.data.assign(4, 0.0f);
  Tensor db; db.shape = {4}; db.data.assign(4, 0.0f);
  layer_norm_backward(x, gamma, upstream, dx, dg, db);
  for (size_t i = 0; i < x.data.size(); ++i) {
    const float numeric = finite_difference(x, i, [&] { return objective(layer_norm(x, gamma, beta), upstream); });
    CHECK(close_grad(dx.data[i], numeric, 3.0e-3f, 3.0e-2f));
  }
  for (size_t i = 0; i < gamma.data.size(); ++i) {
    const float numeric = finite_difference(gamma, i, [&] { return objective(layer_norm(x, gamma, beta), upstream); });
    CHECK(close_grad(dg.data[i], numeric));
  }
  for (size_t i = 0; i < beta.data.size(); ++i) {
    const float numeric = finite_difference(beta, i, [&] { return objective(layer_norm(x, gamma, beta), upstream); });
    CHECK(close_grad(db.data[i], numeric));
  }
}

void test_gelu_gradient() {
  Tensor x = matrix(1, 5), upstream = matrix(1, 5);
  x.data = {-1.7f, -0.4f, 0.0f, 0.6f, 1.8f};
  upstream.data = {0.3f, -0.5f, 0.7f, 0.2f, -0.1f};
  const Tensor dx = gelu_backward(x, upstream);
  for (size_t i = 0; i < x.data.size(); ++i) {
    const float numeric = finite_difference(x, i, [&] { return objective(gelu(x), upstream); });
    CHECK(close_grad(dx.data[i], numeric, 1.0e-3f, 1.5e-2f));
  }
}

void test_softmax_gradient() {
  Tensor x = matrix(2, 3), upstream = matrix(2, 3);
  x.data = {0.2f, -0.1f, 0.7f, -0.5f, 0.9f, 0.3f};
  upstream.data = {0.3f, -0.4f, 0.8f, -0.1f, 0.5f, 0.2f};
  const Tensor y = softmax_rows(x);
  const Tensor dx = softmax_backward(y, upstream);
  for (size_t i = 0; i < x.data.size(); ++i) {
    const float numeric = finite_difference(x, i, [&] { return objective(softmax_rows(x), upstream); });
    CHECK(close_grad(dx.data[i], numeric));
  }
  for (uint32_t r = 0; r < 2; ++r) {
    float sum = 0.0f;
    for (uint32_t c = 0; c < 3; ++c) sum += y.data[r * 3 + c];
    CHECK(std::fabs(sum - 1.0f) < 1.0e-6f);
  }
}

Model tiny_model() {
  Config cfg;
  cfg.vocab_size = 5; cfg.n_layers = 1; cfg.d_model = 4;
  cfg.n_heads = 2; cfg.d_ff = 7; cfg.ctx_len = 4;
  Tokenizer tok;
  tok.tokens = {"a", "b", "c", "d", "e"};
  Model model = make_empty_model(cfg, tok);
  initialize_weights(model, 2026);
  return model;
}
float model_loss(const Model &m, const std::vector<uint32_t> &ids,
                 const std::vector<uint32_t> &targets) {
  Model grads;
  return loss_and_backward(m, ids, targets, grads);
}
void test_transformer_gradient() {
  const std::vector<uint32_t> ids = {0, 1, 2};
  const std::vector<uint32_t> targets = {1, 2, 3};
  Model model = tiny_model(), grads;
  const float loss = loss_and_backward(model, ids, targets, grads);
  CHECK(std::isfinite(loss) && loss > 0.0f);
  CHECK(forward_logits(model, ids).size() == ids.size() * model.cfg.vocab_size);

  const std::vector<std::pair<std::string, size_t>> samples = {
      {"tok_emb", 3}, {"pos_emb", 5}, {"layer0.attn.wqkv", 6},
      {"layer0.attn.wo", 2}, {"layer0.mlp.w1", 9},
      {"layer0.mlp.w2", 4}, {"lnf.g", 1}};
  for (const auto &sample : samples) {
    size_t pi = 0;
    while (model.tensors[pi].name != sample.first) ++pi;
    const float numeric = finite_difference(model.tensors[pi], sample.second,
        [&] { return model_loss(model, ids, targets); });
    const float analytic = grads.tensors[pi].data[sample.second];
    CHECK(close_grad(analytic, numeric, 5.0e-3f, 5.0e-2f));
  }
}

void test_adamw() {
  Model model = tiny_model(), grads;
  const std::vector<uint32_t> ids = {0, 1, 2}, targets = {1, 2, 3};
  const float before = model_loss(model, ids, targets);
  loss_and_backward(model, ids, targets, grads);
  AdamWConfig cfg;
  cfg.learning_rate = 0.02f; cfg.weight_decay = 0.0f; cfg.max_grad_norm = 0.05f;
  AdamW optimizer(model, cfg);
  const OptimizerStats stats = optimizer.step(model, grads);
  const float after = model_loss(model, ids, targets);
  CHECK(stats.step == 1 && optimizer.step_count() == 1);
  CHECK(stats.grad_norm > 0.0 && stats.clip_scale < 1.0f);
  CHECK(after < before);
}

} // namespace

int main() {
  test_matmul_gradient();
  test_layer_norm_gradient();
  test_gelu_gradient();
  test_softmax_gradient();
  test_transformer_gradient();
  test_adamw();
  std::printf("ZIKA core: %d verificacoes, %d falhas\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

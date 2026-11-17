#include "zika/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace zika::ops {
namespace {
void require_matrix(const Tensor &x, const char *name) {
  if (x.shape.size() != 2 || x.data.size() != x.numel())
    throw std::invalid_argument(std::string(name) + " tem de ser uma matriz 2D valida");
}
void require_vector(const Tensor &x, uint32_t n, const char *name) {
  if (x.shape.size() != 1 || x.shape[0] != n || x.data.size() != n)
    throw std::invalid_argument(std::string(name) + " tem dimensoes invalidas");
}
void require_same(const Tensor &a, const Tensor &b, const char *what) {
  if (a.shape != b.shape || a.data.size() != b.data.size())
    throw std::invalid_argument(std::string(what) + ": dimensoes incompativeis");
}
constexpr float kSqrt2OverPi = 0.7978845608028654f;
}

Tensor matrix(uint32_t rows, uint32_t cols, float value) {
  Tensor x;
  x.shape = {rows, cols};
  x.data.assign(static_cast<size_t>(rows) * cols, value);
  return x;
}

Tensor transpose(const Tensor &x) {
  require_matrix(x, "transpose");
  const uint32_t rows = x.shape[0], cols = x.shape[1];
  Tensor y = matrix(cols, rows);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < cols; ++c)
      y.data[static_cast<size_t>(c) * rows + r] =
          x.data[static_cast<size_t>(r) * cols + c];
  return y;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  require_matrix(a, "matmul(a)");
  require_matrix(b, "matmul(b)");
  const uint32_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  if (b.shape[0] != k)
    throw std::invalid_argument("matmul: dimensoes internas incompativeis");
  Tensor c = matrix(m, n);
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t p = 0; p < k; ++p) {
      const float av = a.data[static_cast<size_t>(i) * k + p];
      const size_t brow = static_cast<size_t>(p) * n;
      const size_t crow = static_cast<size_t>(i) * n;
      for (uint32_t j = 0; j < n; ++j)
        c.data[crow + j] += av * b.data[brow + j];
    }
  }
  return c;
}

void matmul_backward(const Tensor &a, const Tensor &b, const Tensor &dy,
                     Tensor &da, Tensor &db) {
  require_matrix(a, "matmul_backward(a)");
  require_matrix(b, "matmul_backward(b)");
  require_matrix(dy, "matmul_backward(dy)");
  const uint32_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  if (b.shape[0] != k || dy.shape[0] != m || dy.shape[1] != n)
    throw std::invalid_argument("matmul_backward: dimensoes incompativeis");
  if (da.shape != a.shape) da = matrix(m, k);
  if (db.shape != b.shape) db = matrix(k, n);
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t p = 0; p < k; ++p) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < n; ++j)
        sum += dy.data[static_cast<size_t>(i) * n + j] *
               b.data[static_cast<size_t>(p) * n + j];
      da.data[static_cast<size_t>(i) * k + p] += sum;
    }
  }
  for (uint32_t p = 0; p < k; ++p) {
    for (uint32_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (uint32_t i = 0; i < m; ++i)
        sum += a.data[static_cast<size_t>(i) * k + p] *
               dy.data[static_cast<size_t>(i) * n + j];
      db.data[static_cast<size_t>(p) * n + j] += sum;
    }
  }
}

Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                  float epsilon) {
  require_matrix(x, "layer_norm(x)");
  const uint32_t rows = x.shape[0], cols = x.shape[1];
  require_vector(gamma, cols, "layer_norm(gamma)");
  require_vector(beta, cols, "layer_norm(beta)");
  if (epsilon <= 0.0f) throw std::invalid_argument("epsilon tem de ser positivo");
  Tensor y = matrix(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * cols;
    float mean = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) mean += x.data[base + c];
    mean /= static_cast<float>(cols);
    float variance = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      const float d = x.data[base + c] - mean;
      variance += d * d;
    }
    variance /= static_cast<float>(cols);
    const float inv = 1.0f / std::sqrt(variance + epsilon);
    for (uint32_t c = 0; c < cols; ++c) {
      const float normalized = (x.data[base + c] - mean) * inv;
      y.data[base + c] = normalized * gamma.data[c] + beta.data[c];
    }
  }
  return y;
}

void layer_norm_backward(const Tensor &x, const Tensor &gamma,
                         const Tensor &dy, Tensor &dx, Tensor &dgamma,
                         Tensor &dbeta, float epsilon) {
  require_matrix(x, "layer_norm_backward(x)");
  require_matrix(dy, "layer_norm_backward(dy)");
  require_same(x, dy, "layer_norm_backward");
  const uint32_t rows = x.shape[0], cols = x.shape[1];
  require_vector(gamma, cols, "layer_norm_backward(gamma)");
  if (epsilon <= 0.0f) throw std::invalid_argument("epsilon tem de ser positivo");
  if (dx.shape != x.shape) dx = matrix(rows, cols);
  if (dgamma.shape != std::vector<uint32_t>{cols}) {
    dgamma.shape = {cols}; dgamma.data.assign(cols, 0.0f);
  }
  if (dbeta.shape != std::vector<uint32_t>{cols}) {
    dbeta.shape = {cols}; dbeta.data.assign(cols, 0.0f);
  }
  for (uint32_t r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * cols;
    float mean = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) mean += x.data[base + c];
    mean /= static_cast<float>(cols);
    float variance = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      const float d = x.data[base + c] - mean;
      variance += d * d;
    }
    variance /= static_cast<float>(cols);
    const float inv = 1.0f / std::sqrt(variance + epsilon);
    float sum_dnorm = 0.0f, sum_dnorm_xhat = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      const float xhat = (x.data[base + c] - mean) * inv;
      const float dnorm = dy.data[base + c] * gamma.data[c];
      sum_dnorm += dnorm;
      sum_dnorm_xhat += dnorm * xhat;
      dgamma.data[c] += dy.data[base + c] * xhat;
      dbeta.data[c] += dy.data[base + c];
    }
    for (uint32_t c = 0; c < cols; ++c) {
      const float xhat = (x.data[base + c] - mean) * inv;
      const float dnorm = dy.data[base + c] * gamma.data[c];
      dx.data[base + c] += (inv / static_cast<float>(cols)) *
                           (static_cast<float>(cols) * dnorm -
                            sum_dnorm - xhat * sum_dnorm_xhat);
    }
  }
}

Tensor gelu(const Tensor &x) {
  Tensor y = x;
  for (float &v : y.data) {
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    v = 0.5f * v * (1.0f + std::tanh(u));
  }
  return y;
}

Tensor gelu_backward(const Tensor &x, const Tensor &dy) {
  require_same(x, dy, "gelu_backward");
  Tensor dx = x;
  for (size_t i = 0; i < x.data.size(); ++i) {
    const float v = x.data[i];
    const float u = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    const float t = std::tanh(u);
    const float du = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * v * v);
    const float derivative = 0.5f * (1.0f + t) +
                             0.5f * v * (1.0f - t * t) * du;
    dx.data[i] = dy.data[i] * derivative;
  }
  return dx;
}

Tensor softmax_rows(const Tensor &x) {
  require_matrix(x, "softmax_rows");
  const uint32_t rows = x.shape[0], cols = x.shape[1];
  Tensor y = matrix(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * cols;
    float maximum = x.data[base];
    for (uint32_t c = 1; c < cols; ++c)
      if (x.data[base + c] > maximum) maximum = x.data[base + c];
    float total = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
      const float e = std::exp(x.data[base + c] - maximum);
      y.data[base + c] = e;
      total += e;
    }
    if (!(total > 0.0f) || !std::isfinite(total))
      throw std::runtime_error("softmax: soma de exponenciais invalida");
    for (uint32_t c = 0; c < cols; ++c) y.data[base + c] /= total;
  }
  return y;
}

Tensor softmax_backward(const Tensor &y, const Tensor &dy) {
  require_matrix(y, "softmax_backward(y)");
  require_matrix(dy, "softmax_backward(dy)");
  require_same(y, dy, "softmax_backward");
  const uint32_t rows = y.shape[0], cols = y.shape[1];
  Tensor dx = matrix(rows, cols);
  for (uint32_t r = 0; r < rows; ++r) {
    const size_t base = static_cast<size_t>(r) * cols;
    float dot = 0.0f;
    for (uint32_t c = 0; c < cols; ++c)
      dot += y.data[base + c] * dy.data[base + c];
    for (uint32_t c = 0; c < cols; ++c)
      dx.data[base + c] = y.data[base + c] * (dy.data[base + c] - dot);
  }
  return dx;
}

} // namespace zika::ops

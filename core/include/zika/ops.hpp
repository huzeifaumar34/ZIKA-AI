// Operacoes numericas de matrizes e os seus gradientes manuais.
#pragma once

#include "zika/format.hpp"

namespace zika::ops {

// Matrizes densas row-major representadas pelo Tensor do formato .zika.
Tensor matrix(uint32_t rows, uint32_t cols, float value = 0.0f);
Tensor transpose(const Tensor &x);
Tensor matmul(const Tensor &a, const Tensor &b);
void matmul_backward(const Tensor &a, const Tensor &b, const Tensor &dy,
                     Tensor &da, Tensor &db);

Tensor layer_norm(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                  float epsilon = 1.0e-5f);
void layer_norm_backward(const Tensor &x, const Tensor &gamma,
                         const Tensor &dy, Tensor &dx, Tensor &dgamma,
                         Tensor &dbeta, float epsilon = 1.0e-5f);

Tensor gelu(const Tensor &x);
Tensor gelu_backward(const Tensor &x, const Tensor &dy);

Tensor softmax_rows(const Tensor &x);
Tensor softmax_backward(const Tensor &y, const Tensor &dy);

} // namespace zika::ops

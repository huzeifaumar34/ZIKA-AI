// Forward e backpropagation do Transformer decoder-only do ZIKA-AI.
#pragma once

#include "zika/format.hpp"

#include <cstdint>
#include <vector>

namespace zika {

// Inicializacao deterministica dos pesos para treino (nao altera o formato .zika).
void initialize_weights(Model &model, uint32_t seed = 1337);

// Logits row-major [numero_de_tokens, vocab_size].
std::vector<float> forward_logits(const Model &model,
                                  const std::vector<uint32_t> &input_ids);

// Cross-entropy causal e gradientes completos dos parametros.
// input_ids e targets têm o mesmo tamanho: cada posição prevê o token em targets.
float loss_and_backward(const Model &model,
                        const std::vector<uint32_t> &input_ids,
                        const std::vector<uint32_t> &targets,
                        Model &gradients);

} // namespace zika

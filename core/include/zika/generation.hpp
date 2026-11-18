// Tokenizacao em tempo de inferencia e geracao autoregressiva.
#pragma once

#include "zika/format.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace zika {

struct GenerationConfig {
  uint32_t max_new_tokens = 160;
  float temperature = 0.8f;
  uint32_t top_k = 40; // 0 desativa top-k
  float top_p = 0.9f;  // 1.0 desativa nucleus filtering
};

std::string encode_text(const Tokenizer &tokenizer, const std::string &text,
                        std::vector<uint32_t> &ids);
std::string decode_tokens(const Tokenizer &tokenizer,
                          const std::vector<uint32_t> &ids,
                          std::string &text, bool skip_special = true);

uint32_t sample_next_token(const std::vector<float> &logits,
                           const GenerationConfig &config,
                           std::mt19937 &rng);

// O contexto inclui os marcadores de papel. A funcao devolve apenas tokens de
// texto; os marcadores especiais terminam a resposta e nunca sao apresentados.
std::string generate_tokens(const Model &model,
                            const std::vector<uint32_t> &context,
                            const GenerationConfig &config,
                            std::mt19937 &rng,
                            std::vector<uint32_t> &generated);

} // namespace zika

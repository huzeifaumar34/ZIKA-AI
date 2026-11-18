#include "zika/generation.hpp"

#include "zika/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace zika {
namespace {

size_t utf8_width(unsigned char lead) {
  if (lead <= 0x7F) return 1;
  if (lead >= 0xC2 && lead <= 0xDF) return 2;
  if (lead >= 0xE0 && lead <= 0xEF) return 3;
  if (lead >= 0xF0 && lead <= 0xF4) return 4;
  return 0;
}

bool valid_utf8_unit(const std::string &text, size_t pos, size_t width) {
  if (width == 0 || pos + width > text.size()) return false;
  for (size_t i = 1; i < width; ++i) {
    const unsigned char c = static_cast<unsigned char>(text[pos + i]);
    if ((c & 0xC0) != 0x80) return false;
  }
  if (width == 3) {
    const unsigned char a = static_cast<unsigned char>(text[pos]);
    const unsigned char b = static_cast<unsigned char>(text[pos + 1]);
    if ((a == 0xE0 && b < 0xA0) || (a == 0xED && b >= 0xA0)) return false;
  } else if (width == 4) {
    const unsigned char a = static_cast<unsigned char>(text[pos]);
    const unsigned char b = static_cast<unsigned char>(text[pos + 1]);
    if ((a == 0xF0 && b < 0x90) || (a == 0xF4 && b >= 0x90)) return false;
  }
  return true;
}

bool valid_sampling_config(const GenerationConfig &cfg) {
  return cfg.max_new_tokens > 0 && std::isfinite(cfg.temperature) &&
         cfg.temperature > 0.0f && std::isfinite(cfg.top_p) &&
         cfg.top_p > 0.0f && cfg.top_p <= 1.0f;
}

} // namespace

std::string encode_text(const Tokenizer &tokenizer, const std::string &text,
                        std::vector<uint32_t> &ids) {
  if (tokenizer.kind != TokenizerKind::Char)
    return "apenas o tokenizer de caracteres esta implementado";
  if (tokenizer.tokens.size() < 4)
    return "vocabulario de caracteres invalido";

  std::vector<uint32_t> encoded;
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t width = utf8_width(static_cast<unsigned char>(text[pos]));
    if (!valid_utf8_unit(text, pos, width))
      return "texto contem UTF-8 invalido na posicao " + std::to_string(pos);
    const std::string unit = text.substr(pos, width);
    bool found = false;
    for (uint32_t id = 3; id < tokenizer.tokens.size(); ++id) {
      if (tokenizer.tokens[id] == unit) {
        encoded.push_back(id);
        found = true;
        break;
      }
    }
    if (!found)
      return "caractere nao suportado na posicao " + std::to_string(pos) +
             ": " + unit;
    pos += width;
  }
  ids.insert(ids.end(), encoded.begin(), encoded.end());
  return "";
}

std::string decode_tokens(const Tokenizer &tokenizer,
                          const std::vector<uint32_t> &ids,
                          std::string &text, bool skip_special) {
  if (tokenizer.kind != TokenizerKind::Char)
    return "apenas o tokenizer de caracteres esta implementado";
  std::string decoded;
  for (uint32_t id : ids) {
    if (id >= tokenizer.tokens.size())
      return "ID de token fora do vocabulario: " + std::to_string(id);
    if (skip_special && id < 3) continue;
    decoded += tokenizer.tokens[id];
  }
  text.swap(decoded);
  return "";
}

uint32_t sample_next_token(const std::vector<float> &logits,
                           const GenerationConfig &cfg, std::mt19937 &rng) {
  if (logits.empty()) throw std::invalid_argument("logits vazios");
  if (!valid_sampling_config(cfg))
    throw std::invalid_argument("configuracao de geracao invalida");
  for (float value : logits)
    if (!std::isfinite(value))
      throw std::invalid_argument("logits contem valor nao finito");

  std::vector<uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return logits[a] > logits[b];
  });
  if (cfg.top_k > 0 && cfg.top_k < order.size())
    order.resize(cfg.top_k);

  const float maximum = logits[order.front()] / cfg.temperature;
  std::vector<double> weights;
  weights.reserve(order.size());
  double total = 0.0;
  for (uint32_t id : order) {
    const double weight =
        std::exp(static_cast<double>(logits[id] / cfg.temperature - maximum));
    weights.push_back(weight);
    total += weight;
  }
  if (!(total > 0.0) || !std::isfinite(total))
    throw std::runtime_error("nao foi possivel normalizar os logits");

  if (cfg.top_p < 1.0f && order.size() > 1) {
    size_t keep = 0;
    double cumulative = 0.0;
    do {
      cumulative += weights[keep] / total;
      ++keep;
    } while (keep < weights.size() && cumulative < cfg.top_p);
    order.resize(keep);
    weights.resize(keep);
  }

  std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
  return order[distribution(rng)];
}

std::string generate_tokens(const Model &model,
                            const std::vector<uint32_t> &context,
                            const GenerationConfig &cfg, std::mt19937 &rng,
                            std::vector<uint32_t> &generated) {
  generated.clear();
  if (!valid_sampling_config(cfg))
    return "configuracao de geracao invalida";
  if (context.empty()) return "o contexto nao pode estar vazio";
  if (model.tok.kind != TokenizerKind::Char ||
      model.tok.tokens.size() != model.cfg.vocab_size)
    return "modelo nao usa o tokenizer char esperado";
  if (model.cfg.vocab_size < 4) return "vocabulario do modelo invalido";

  std::vector<uint32_t> running = context;
  for (uint32_t step = 0; step < cfg.max_new_tokens; ++step) {
    const size_t begin = running.size() > model.cfg.ctx_len
                             ? running.size() - model.cfg.ctx_len
                             : 0;
    std::vector<uint32_t> window(running.begin() + begin, running.end());
    std::vector<float> logits;
    try {
      logits = forward_logits(model, window);
    } catch (const std::exception &e) {
      return std::string("falha no forward: ") + e.what();
    }
    const size_t offset = (window.size() - 1) * model.cfg.vocab_size;
    std::vector<float> last(logits.begin() + offset,
                            logits.begin() + offset + model.cfg.vocab_size);
    uint32_t next = 0;
    try {
      next = sample_next_token(last, cfg, rng);
    } catch (const std::exception &e) {
      return std::string("falha na amostragem: ") + e.what();
    }

    // IDs 0, 1 e 2 sao marcadores de estrutura, nunca texto da resposta.
    if (next < 3) break;
    generated.push_back(next);
    running.push_back(next);
  }
  return "";
}

} // namespace zika

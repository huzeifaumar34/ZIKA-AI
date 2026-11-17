#include "zika/generation.hpp"
#include "zika/format.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
bool parse_u32(const std::string &text, uint32_t &value) {
  if (text.empty() || text[0] == '-') return false;
  try {
    size_t used = 0;
    const unsigned long parsed = std::stoul(text, &used, 10);
    if (used != text.size() || parsed > UINT32_MAX) return false;
    value = static_cast<uint32_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_float(const std::string &text, float &value) {
  try {
    size_t used = 0;
    const float parsed = std::stof(text, &used);
    if (used != text.size() || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

void trim_history(std::vector<uint32_t> &history, size_t required,
                  uint32_t context_size) {
  if (required >= context_size) {
    history.clear();
    return;
  }
  while (history.size() + required > context_size) {
    // Preferir iniciar o contexto no marcador de uma nova mensagem do utilizador.
    size_t next_user = 1;
    while (next_user < history.size() && history[next_user] != 0) ++next_user;
    if (next_user < history.size())
      history.erase(history.begin(), history.begin() + next_user);
    else
      history.erase(history.begin());
  }
}
} // namespace

int cmd_chat(int argc, char **argv) {
  std::string model_path;
  zika::GenerationConfig config;
  uint32_t seed = 1337;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" || arg == "--max-new-tokens" || arg == "--temperature" ||
        arg == "--top-k" || arg == "--top-p" || arg == "--seed") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "erro: falta valor para %s\n", arg.c_str());
        return 1;
      }
      const std::string value = argv[++i];
      if (arg == "--model") {
        model_path = value;
      } else if (arg == "--temperature" || arg == "--top-p") {
        float parsed = 0.0f;
        if (!parse_float(value, parsed)) {
          std::fprintf(stderr, "erro: numero invalido para %s\n", arg.c_str());
          return 1;
        }
        if (arg == "--temperature") config.temperature = parsed;
        else config.top_p = parsed;
      } else {
        uint32_t parsed = 0;
        if (!parse_u32(value, parsed)) {
          std::fprintf(stderr, "erro: inteiro invalido para %s\n", arg.c_str());
          return 1;
        }
        if (arg == "--max-new-tokens") config.max_new_tokens = parsed;
        else if (arg == "--top-k") config.top_k = parsed;
        else seed = parsed;
      }
    } else {
      std::fprintf(stderr, "erro: argumento desconhecido: %s\n", arg.c_str());
      return 1;
    }
  }

  if (model_path.empty()) {
    std::fprintf(stderr,
                 "Uso: zika chat --model <modelo.zika> [--max-new-tokens N] "
                 "[--temperature F] [--top-k N] [--top-p F] [--seed N]\n");
    return 1;
  }

  zika::Model model;
  const std::string load_error = zika::read_zika(model_path, model);
  if (!load_error.empty()) {
    std::fprintf(stderr, "erro ao carregar modelo: %s\n", load_error.c_str());
    return 1;
  }
  if (model.tok.kind != zika::TokenizerKind::Char ||
      model.tok.tokens.size() != model.cfg.vocab_size) {
    std::fprintf(stderr, "erro: o chat atual requer tokenizer de caracteres char-v0.1\n");
    return 1;
  }
  if (config.max_new_tokens == 0 || config.temperature <= 0.0f ||
      config.top_p <= 0.0f || config.top_p > 1.0f) {
    std::fprintf(stderr, "erro: max-new-tokens > 0, temperature > 0 e 0 < top-p <= 1\n");
    return 1;
  }

  std::mt19937 rng(seed);
  std::vector<uint32_t> history;
  std::string line;
  std::cout << "ZIKA-AI | modelo: " << model_path << "\n"
            << "Comandos: /sair para terminar.\n";
  while (true) {
    std::cout << "\nTu: " << std::flush;
    if (!std::getline(std::cin, line)) break;
    if (line == "/sair" || line == "/exit" || line == "/quit") break;
    if (line.empty()) continue;

    std::vector<uint32_t> user_ids;
    const std::string encode_error = zika::encode_text(model.tok, line, user_ids);
    if (!encode_error.empty()) {
      std::cout << "ZIKA-AI: nao consegui codificar a mensagem: "
                << encode_error << "\n";
      continue;
    }
    if (user_ids.size() + 2 > model.cfg.ctx_len) {
      std::cout << "ZIKA-AI: mensagem demasiado longa. Limite: "
                << (model.cfg.ctx_len - 2) << " caracteres/token(s).\n";
      continue;
    }

    trim_history(history, user_ids.size() + 2, model.cfg.ctx_len);
    history.push_back(0); // <|user|>
    history.insert(history.end(), user_ids.begin(), user_ids.end());
    history.push_back(1); // <|ia|>

    std::vector<uint32_t> answer_ids;
    const std::string generation_error =
        zika::generate_tokens(model, history, config, rng, answer_ids);
    if (!generation_error.empty()) {
      std::cout << "ZIKA-AI: erro de geracao: " << generation_error << "\n";
      history.clear();
      continue;
    }

    std::string answer;
    const std::string decode_error =
        zika::decode_tokens(model.tok, answer_ids, answer, true);
    if (!decode_error.empty()) {
      std::cout << "ZIKA-AI: erro ao descodificar resposta: "
                << decode_error << "\n";
      history.clear();
      continue;
    }
    std::cout << "ZIKA: " << (answer.empty() ? "[sem resposta]" : answer) << "\n";
    history.insert(history.end(), answer_ids.begin(), answer_ids.end());
  }
  std::cout << "\nSessao terminada.\n";
  return 0;
}

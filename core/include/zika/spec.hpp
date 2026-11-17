// Leitura dos ficheiros de configuração (configs/*.json).
#pragma once

#include <string>

#include "zika/format.hpp"

namespace zika {

struct ModelSpec {
  std::string name;
  std::string tokenizer = "char";
  Config cfg; // vocab_size fica a 0: é decidido pelo tokenizer
};

// JSON plano: um objeto com chaves de texto e valores inteiros ou texto.
// Chaves: name, tokenizer, n_layers, d_model, n_heads, d_ff, ctx_len.
// Devolve "" em caso de sucesso, ou a descrição do erro.
std::string parse_spec_json(const std::string &text, ModelSpec &spec);
std::string load_spec_json(const std::string &path, ModelSpec &spec);

} // namespace zika

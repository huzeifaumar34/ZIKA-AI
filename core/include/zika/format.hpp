// Formato de ficheiro .zika (versão 1) e estruturas do modelo.
//
// Disposição (tudo em little-endian):
//   "ZIKA" | u32 versão | u32 dtype
//   u32 vocab_size, n_layers, d_model, n_heads, d_ff, ctx_len
//   u32 tipo_tokenizer | u32 n_tokens | { u32 len, bytes }*
//   u32 n_merges | { u32 a, u32 b }*
//   u32 n_tensores | { u32 len_nome, nome, u32 ndim, u32 dims[ndim], dados }*
//   u32 crc32 (de todos os bytes anteriores)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace zika {

constexpr uint32_t kFormatVersion = 1;

// Só F32 está implementado; F16 e I8 ficam reservados (fase 7).
enum class DType : uint32_t { F32 = 0, F16 = 1, I8 = 2 };

// BPE fica reservado para a fase 3.
enum class TokenizerKind : uint32_t { Char = 0, BPE = 1 };

struct Config {
  uint32_t vocab_size = 0;
  uint32_t n_layers = 0;
  uint32_t d_model = 0;
  uint32_t n_heads = 0;
  uint32_t d_ff = 0;
  uint32_t ctx_len = 0;
};

struct Tokenizer {
  TokenizerKind kind = TokenizerKind::Char;
  std::vector<std::string> tokens;                   // id -> bytes do token
  std::vector<std::pair<uint32_t, uint32_t>> merges; // regras BPE (fase 3)
};

struct Tensor {
  std::string name;
  std::vector<uint32_t> shape;
  std::vector<float> data;
  uint64_t numel() const;
};

struct Model {
  Config cfg;
  DType dtype = DType::F32;
  Tokenizer tok;
  std::vector<Tensor> tensors; // pela ordem de tensor_specs(cfg)
};

struct TensorSpec {
  std::string name;
  std::vector<uint32_t> shape;
};

// Devolve "" se a configuração é válida, ou a descrição do problema.
std::string validate_config(const Config &cfg);

// Lista (nome, forma) de todos os tensores do modelo, pela ordem do ficheiro.
// A camada de saída partilha os pesos de tok_emb.
std::vector<TensorSpec> tensor_specs(const Config &cfg);

uint64_t param_count(const Config &cfg);

// Vocabulário letra a letra: tokens especiais, quebra de linha, ASCII
// imprimível e as letras acentuadas do português.
std::vector<std::string> default_char_vocab();

// Modelo com todos os pesos a zero (ganhos das LayerNorm a 1).
Model make_empty_model(const Config &cfg, const Tokenizer &tok);

uint32_t crc32(const uint8_t *data, size_t size);

// Todas as funções devolvem "" em caso de sucesso, ou a descrição do erro.
std::vector<uint8_t> serialize(const Model &model, std::string &err);
std::string deserialize(const uint8_t *data, size_t size, Model &out);
std::string write_zika(const Model &model, const std::string &path);
std::string read_zika(const std::string &path, Model &out);

} // namespace zika

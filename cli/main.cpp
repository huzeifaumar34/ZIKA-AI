#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "zika/format.hpp"
#include "zika/spec.hpp"
#include "zika/training.hpp"
#include "zika/transformer.hpp"

int cmd_chat(int argc, char **argv);

#ifndef ZIKA_VERSION
#define ZIKA_VERSION "dev"
#endif

using namespace zika;

namespace {

void usage() {
  std::fprintf(stderr,
               "ZIKA-AI %s\n\n"
               "Uso:\n"
               "  zika version\n"
               "  zika init --config <ficheiro.json> --out <modelo.zika>\n"
               "  zika info <modelo.zika> [--tensors]\n"
               "  zika train --config <ficheiro.json> --data <train.bin>\n"
               "      --val-data <val.bin> --out <modelo.zika> [opcoes]\n"
               "  zika chat --model <modelo.zika> [opcoes]\n\n"
               "Opcoes de chat:\n"
               "  --max-new-tokens N     maximo de tokens por resposta (default 160)\n"
               "  --temperature F        temperatura (default 0.8)\n"
               "  --top-k N              candidatos por passo; 0 desativa (default 40)\n"
               "  --top-p F              nucleus sampling (default 0.9)\n"
               "  --seed N               semente da amostragem (default 1337)\n\n"
               "Opcoes de treino:\n"
               "  --steps N              passos totais (default 500)\n"
               "  --batch-size N         sequencias por passo (default 1)\n"
               "  --seq-len N            tokens por sequencia (default 128)\n"
               "  --learning-rate F      taxa maxima (default 0.0003)\n"
               "  --warmup-steps N       passos de aquecimento (default 50)\n"
               "  --weight-decay F       decaimento AdamW (default 0.01)\n"
               "  --grad-clip F          norma maxima (default 1.0; 0 desativa)\n"
               "  --min-lr-ratio F       LR final / LR maxima (default 0.1)\n"
               "  --seed N               semente (default 1337)\n"
               "  --log-every N          intervalo de log (default 25; 0 desativa)\n"
               "  --checkpoint-every N   intervalo de checkpoint (default 100)\n"
               "  --checkpoint PATH      ficheiro de checkpoint\n"
               "  --resume PATH          retomar de checkpoint\n",
               ZIKA_VERSION);
}

int fail(const std::string &msg) {
  std::fprintf(stderr, "erro: %s\n", msg.c_str());
  return 1;
}

bool parse_u64(const std::string &text, uint64_t &value) {
  if (text.empty() || text[0] == '-') return false;
  try {
    size_t used = 0;
    const unsigned long long parsed = std::stoull(text, &used, 10);
    if (used != text.size()) return false;
    value = static_cast<uint64_t>(parsed);
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

std::string human_size(uint64_t bytes) {
  char buf[64];
  if (bytes >= (1u << 20))
    std::snprintf(buf, sizeof buf, "%.2f MiB", bytes / 1048576.0);
  else
    std::snprintf(buf, sizeof buf, "%.1f KiB", bytes / 1024.0);
  return buf;
}

int cmd_init(int argc, char **argv) {
  std::string config_path, out_path;
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    if ((a == "--config" || a == "--out") && i + 1 < argc)
      (a == "--config" ? config_path : out_path) = argv[++i];
    else
      return fail("argumento desconhecido ou sem valor: " + a);
  }
  if (config_path.empty() || out_path.empty()) {
    usage();
    return 1;
  }

  ModelSpec spec;
  std::string err = load_spec_json(config_path, spec);
  if (!err.empty()) return fail(config_path + ": " + err);

  Tokenizer tok;
  tok.kind = TokenizerKind::Char;
  tok.tokens = default_char_vocab();
  spec.cfg.vocab_size = static_cast<uint32_t>(tok.tokens.size());
  err = validate_config(spec.cfg);
  if (!err.empty()) return fail(config_path + ": " + err);

  const Model model = make_empty_model(spec.cfg, tok);
  const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) return fail("nao consegui criar a pasta " + parent.string());
  }
  err = write_zika(model, out_path);
  if (!err.empty()) return fail(err);

  std::printf("modelo '%s' criado em %s\n", spec.name.c_str(), out_path.c_str());
  std::printf("  parametros: %llu (pesos a zero; usa 'zika train' para treinar)\n",
              static_cast<unsigned long long>(param_count(spec.cfg)));
  return 0;
}

int cmd_info(int argc, char **argv) {
  std::string path;
  bool show_tensors = false;
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--tensors") show_tensors = true;
    else if (path.empty()) path = a;
    else return fail("argumento a mais: " + a);
  }
  if (path.empty()) {
    usage();
    return 1;
  }

  Model m;
  const std::string err = read_zika(path, m);
  if (!err.empty()) return fail(path + ": " + err);

  const char *dtype = "F32";
  const char *tok_kind = m.tok.kind == TokenizerKind::Char ? "caracteres" : "BPE";
  std::printf("ficheiro:    %s (%s)\n", path.c_str(),
              human_size(std::filesystem::file_size(path)).c_str());
  std::printf("formato:     .zika v%u, pesos %s\n", kFormatVersion, dtype);
  std::printf("camadas:     %u\n", m.cfg.n_layers);
  std::printf("dimensao:    %u (%u cabecas, d_ff %u)\n", m.cfg.d_model,
              m.cfg.n_heads, m.cfg.d_ff);
  std::printf("contexto:    %u\n", m.cfg.ctx_len);
  std::printf("tokenizer:   %s, %u tokens, %zu regras\n", tok_kind,
              m.cfg.vocab_size, m.tok.merges.size());
  std::printf("parametros:  %llu\n",
              static_cast<unsigned long long>(param_count(m.cfg)));
  std::printf("tensores:    %zu\n", m.tensors.size());
  if (show_tensors) {
    for (const Tensor &t : m.tensors) {
      std::printf("  %-22s [", t.name.c_str());
      for (size_t k = 0; k < t.shape.size(); k++)
        std::printf(k ? ", %u" : "%u", t.shape[k]);
      std::printf("]\n");
    }
  }
  return 0;
}

int cmd_train(int argc, char **argv) {
  std::string config_path, train_path, validation_path, out_path;
  std::string checkpoint_path, resume_path;
  TrainingConfig cfg;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" || arg == "--data" || arg == "--val-data" ||
        arg == "--out" || arg == "--checkpoint" || arg == "--resume") {
      if (i + 1 >= argc) return fail("falta valor para " + arg);
      const std::string value = argv[++i];
      if (arg == "--config") config_path = value;
      else if (arg == "--data") train_path = value;
      else if (arg == "--val-data") validation_path = value;
      else if (arg == "--out") out_path = value;
      else if (arg == "--checkpoint") checkpoint_path = value;
      else resume_path = value;
      continue;
    }
    if (arg == "--steps" || arg == "--batch-size" || arg == "--seq-len" ||
        arg == "--warmup-steps" || arg == "--seed" || arg == "--log-every" ||
        arg == "--checkpoint-every") {
      if (i + 1 >= argc) return fail("falta valor para " + arg);
      uint64_t value = 0;
      if (!parse_u64(argv[++i], value)) return fail("inteiro invalido para " + arg);
      if (arg == "--steps") cfg.steps = value;
      else if (arg == "--batch-size" && value <= UINT32_MAX) cfg.batch_size = static_cast<uint32_t>(value);
      else if (arg == "--seq-len" && value <= UINT32_MAX) cfg.sequence_length = static_cast<uint32_t>(value);
      else if (arg == "--warmup-steps") cfg.warmup_steps = value;
      else if (arg == "--seed" && value <= UINT32_MAX) cfg.seed = static_cast<uint32_t>(value);
      else if (arg == "--log-every" && value <= UINT32_MAX) cfg.log_every = static_cast<uint32_t>(value);
      else if (arg == "--checkpoint-every") cfg.checkpoint_every = value;
      else return fail("valor fora do intervalo para " + arg);
      continue;
    }
    if (arg == "--learning-rate" || arg == "--weight-decay" ||
        arg == "--grad-clip" || arg == "--min-lr-ratio") {
      if (i + 1 >= argc) return fail("falta valor para " + arg);
      float value = 0.0f;
      if (!parse_float(argv[++i], value)) return fail("numero invalido para " + arg);
      if (arg == "--learning-rate") cfg.learning_rate = value;
      else if (arg == "--weight-decay") cfg.weight_decay = value;
      else if (arg == "--grad-clip") cfg.max_grad_norm = value;
      else cfg.min_learning_rate_ratio = value;
      continue;
    }
    return fail("argumento desconhecido: " + arg);
  }

  if (config_path.empty() || train_path.empty() ||
      validation_path.empty() || out_path.empty()) {
    usage();
    return 1;
  }
  if (checkpoint_path.empty())
    checkpoint_path = resume_path.empty() ? out_path + ".ckpt" : resume_path;

  ModelSpec spec;
  std::string err = load_spec_json(config_path, spec);
  if (!err.empty()) return fail(config_path + ": " + err);
  Tokenizer tok;
  tok.kind = TokenizerKind::Char;
  tok.tokens = default_char_vocab();
  spec.cfg.vocab_size = static_cast<uint32_t>(tok.tokens.size());
  err = validate_config(spec.cfg);
  if (!err.empty()) return fail(config_path + ": " + err);

  std::vector<uint32_t> train_tokens, validation_tokens;
  err = load_token_ids(train_path, spec.cfg.vocab_size, train_tokens);
  if (!err.empty()) return fail(err);
  err = load_token_ids(validation_path, spec.cfg.vocab_size, validation_tokens);
  if (!err.empty()) return fail(err);

  Model model = make_empty_model(spec.cfg, tok);
  initialize_weights(model, cfg.seed);
  const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) return fail("nao consegui criar a pasta " + parent.string());
  }

  std::printf("ZIKA-AI: treino de %s\n", spec.name.c_str());
  std::printf("  parametros: %llu | treino: %zu tokens | validacao: %zu tokens\n",
              static_cast<unsigned long long>(param_count(spec.cfg)),
              train_tokens.size(), validation_tokens.size());
  std::printf("  steps: %llu | batch: %u | seq_len: %u | seed: %u\n",
              static_cast<unsigned long long>(cfg.steps), cfg.batch_size,
              cfg.sequence_length, cfg.seed);

  TrainingResult result;
  err = train_model(model, train_tokens, validation_tokens, cfg,
                    out_path, checkpoint_path, resume_path, result);
  if (!err.empty()) return fail(err);
  std::printf("modelo guardado: %s\ncheckpoint: %s\n",
              out_path.c_str(), checkpoint_path.c_str());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 1;
  }
  const std::string cmd = argv[1];
  if (cmd == "version" || cmd == "--version") {
    std::printf("zika %s\n", ZIKA_VERSION);
    return 0;
  }
  if (cmd == "init") return cmd_init(argc, argv);
  if (cmd == "info") return cmd_info(argc, argv);
  if (cmd == "train") return cmd_train(argc, argv);
  if (cmd == "chat") return cmd_chat(argc, argv);
  if (cmd == "help" || cmd == "--help" || cmd == "-h") {
    usage();
    return 0;
  }
  usage();
  return 1;
}

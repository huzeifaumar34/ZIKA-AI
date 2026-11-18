#include <algorithm>
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
      "  zika eval --model <modelo.zika> --data <test.bin> [opcoes]\n\n"
      "Opcoes de treino:\n"
      "  --steps N                   passo total pretendido (default 500)\n"
      "  --batch-size N              sequencias por passo (default 1)\n"
      "  --seq-len N                 tokens por sequencia (default 128)\n"
      "  --learning-rate F           taxa maxima (default 0.0003)\n"
      "  --warmup-steps N            passos iniciais de aquecimento\n"
      "  --weight-decay F            decaimento AdamW (default 0.01)\n"
      "  --grad-clip F               norma maxima (default 1.0; 0 desativa)\n"
      "  --min-lr-ratio F            LR final / LR maxima (default 0.1)\n"
      "  --log-every N               intervalo de log (default 25; 0 desativa)\n"
      "  --eval-every N              avaliacao amostrada (default 100; 0 desativa)\n"
      "  --eval-tokens N             maximo de tokens por avaliacao (default 2048)\n"
      "  --full-eval-every N         avaliacao completa (default 500; 0 desativa)\n"
      "  --resume-warmup-steps N     rampa ao prolongar treino concluido (default 50)\n"
      "  --resume-legacy-total-steps N  horizonte original para checkpoint ZCKP v1\n"
      "  --max-steps-this-run N       pausa apos N passos e guarda o checkpoint\n"
      "  --checkpoint PATH           ficheiro de checkpoint\n"
      "  --resume PATH               retomar de checkpoint\n\n"
      "Opcoes de avaliacao:\n"
      "  --model PATH                modelo .zika treinado\n"
      "  --data PATH                 tokens uint32 little-endian\n"
      "  --seq-len N                 comprimento maximo do contexto (default 128)\n"
      "  --max-tokens N              0 avalia todos; N limita a amostra (default 0)\n",
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
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--config" || arg == "--out") && i + 1 < argc)
      (arg == "--config" ? config_path : out_path) = argv[++i];
    else
      return fail("argumento desconhecido ou sem valor: " + arg);
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
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tensors") show_tensors = true;
    else if (path.empty()) path = arg;
    else return fail("argumento a mais: " + arg);
  }
  if (path.empty()) {
    usage();
    return 1;
  }

  Model model;
  const std::string err = read_zika(path, model);
  if (!err.empty()) return fail(path + ": " + err);
  const char *tok_kind = model.tok.kind == TokenizerKind::Char ? "caracteres" : "BPE";
  std::printf("ficheiro:    %s (%s)\n", path.c_str(),
      human_size(std::filesystem::file_size(path)).c_str());
  std::printf("formato:     .zika v%u, pesos F32\n", kFormatVersion);
  std::printf("camadas:     %u\n", model.cfg.n_layers);
  std::printf("dimensao:    %u (%u cabecas, d_ff %u)\n",
      model.cfg.d_model, model.cfg.n_heads, model.cfg.d_ff);
  std::printf("contexto:    %u\n", model.cfg.ctx_len);
  std::printf("tokenizer:   %s, %u tokens, %zu regras\n",
      tok_kind, model.cfg.vocab_size, model.tok.merges.size());
  std::printf("parametros:  %llu\n",
      static_cast<unsigned long long>(param_count(model.cfg)));
  std::printf("tensores:    %zu\n", model.tensors.size());
  if (show_tensors) {
    for (const Tensor &tensor : model.tensors) {
      std::printf("  %-22s [", tensor.name.c_str());
      for (size_t k = 0; k < tensor.shape.size(); ++k)
        std::printf(k ? ", %u" : "%u", tensor.shape[k]);
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
        arg == "--checkpoint-every" || arg == "--eval-every" ||
        arg == "--eval-tokens" || arg == "--full-eval-every" ||
        arg == "--resume-warmup-steps" || arg == "--resume-legacy-total-steps" ||
        arg == "--max-steps-this-run") {
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
      else if (arg == "--eval-every" && value <= UINT32_MAX) cfg.eval_every = static_cast<uint32_t>(value);
      else if (arg == "--eval-tokens" && value <= UINT32_MAX) cfg.eval_tokens = static_cast<uint32_t>(value);
      else if (arg == "--full-eval-every") cfg.full_eval_every = value;
      else if (arg == "--resume-warmup-steps" && value <= UINT32_MAX) cfg.resume_warmup_steps = static_cast<uint32_t>(value);
      else if (arg == "--resume-legacy-total-steps") cfg.legacy_resume_total_steps = value;
      else if (arg == "--max-steps-this-run") cfg.max_steps_this_run = value;
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

int cmd_eval(int argc, char **argv) {
  std::string model_path, data_path;
  uint64_t sequence_length = 128, max_tokens = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" || arg == "--data" ||
        arg == "--seq-len" || arg == "--max-tokens") {
      if (i + 1 >= argc) return fail("falta valor para " + arg);
      const std::string value = argv[++i];
      if (arg == "--model") model_path = value;
      else if (arg == "--data") data_path = value;
      else {
        uint64_t parsed = 0;
        if (!parse_u64(value, parsed)) return fail("inteiro invalido para " + arg);
        if (arg == "--seq-len") sequence_length = parsed;
        else max_tokens = parsed;
      }
    } else return fail("argumento desconhecido: " + arg);
  }
  if (model_path.empty() || data_path.empty() || sequence_length == 0 ||
      sequence_length > UINT32_MAX)
    return fail("use --model, --data e --seq-len valido");

  Model model;
  std::string err = read_zika(model_path, model);
  if (!err.empty()) return fail(model_path + ": " + err);
  std::vector<uint32_t> tokens;
  err = load_token_ids(data_path, model.cfg.vocab_size, tokens);
  if (!err.empty()) return fail(err);
  try {
    const float loss = evaluate_loss(model, tokens,
        static_cast<uint32_t>(sequence_length), max_tokens);
    std::printf("eval_loss %.6f | tokens %zu | modo %s\n",
        loss, max_tokens == 0 ? tokens.size() - 1 :
        std::min<uint64_t>(max_tokens, tokens.size() - 1),
        max_tokens == 0 ? "completo" : "amostrado");
  } catch (const std::exception &ex) {
    return fail(ex.what());
  }
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
  if (cmd == "eval") return cmd_eval(argc, argv);
  if (cmd == "help" || cmd == "--help" || cmd == "-h") {
    usage();
    return 0;
  }
  usage();
  return 1;
}

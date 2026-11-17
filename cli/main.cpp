#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "zika/format.hpp"
#include "zika/spec.hpp"

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
               "  zika info <modelo.zika> [--tensors]\n",
               ZIKA_VERSION);
}

int fail(const std::string &msg) {
  std::fprintf(stderr, "erro: %s\n", msg.c_str());
  return 1;
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
  if (!err.empty())
    return fail(config_path + ": " + err);

  Tokenizer tok;
  tok.kind = TokenizerKind::Char;
  tok.tokens = default_char_vocab();
  spec.cfg.vocab_size = static_cast<uint32_t>(tok.tokens.size());
  err = validate_config(spec.cfg);
  if (!err.empty())
    return fail(config_path + ": " + err);

  const Model model = make_empty_model(spec.cfg, tok);

  const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec)
      return fail("nao consegui criar a pasta " + parent.string());
  }
  err = write_zika(model, out_path);
  if (!err.empty())
    return fail(err);

  std::printf("modelo '%s' criado em %s\n", spec.name.c_str(), out_path.c_str());
  std::printf("  parametros: %llu (pesos a zero; o treino vem na fase 4)\n",
              static_cast<unsigned long long>(param_count(spec.cfg)));
  return 0;
}

int cmd_info(int argc, char **argv) {
  std::string path;
  bool show_tensors = false;
  for (int i = 2; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--tensors")
      show_tensors = true;
    else if (path.empty())
      path = a;
    else
      return fail("argumento a mais: " + a);
  }
  if (path.empty()) {
    usage();
    return 1;
  }

  Model m;
  const std::string err = read_zika(path, m);
  if (!err.empty())
    return fail(path + ": " + err);

  const char *dtype = "F32";
  const char *tok_kind = m.tok.kind == TokenizerKind::Char ? "letras" : "BPE";
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
  if (cmd == "init")
    return cmd_init(argc, argv);
  if (cmd == "info")
    return cmd_info(argc, argv);
  if (cmd == "help" || cmd == "--help" || cmd == "-h") {
    usage();
    return 0;
  }
  usage();
  return 1;
}

#include "zika/training.hpp"

#include "zika/optimizer.hpp"
#include "zika/transformer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace zika {
namespace {

struct Writer {
  std::vector<uint8_t> bytes;
  void u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void raw(const uint8_t *p, size_t n) {
    if (n) bytes.insert(bytes.end(), p, p + n);
  }
};
struct Reader {
  const uint8_t *p;
  const uint8_t *end;
  size_t left() const { return static_cast<size_t>(end - p); }
  bool u32(uint32_t &v) {
    if (left() < 4) return false;
    v = static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return true;
  }
  bool u64(uint64_t &v) {
    if (left() < 8) return false;
    v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    p += 8;
    return true;
  }
  bool raw(std::vector<uint8_t> &out, size_t n) {
    if (left() < n) return false;
    out.assign(p, p + n);
    p += n;
    return true;
  }
};

std::string read_bytes(const std::string &path, std::vector<uint8_t> &out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return "nao consegui abrir " + path;
  const std::streamoff end = f.tellg();
  if (end < 0 || static_cast<uint64_t>(end) > (1ull << 32))
    return "tamanho de ficheiro invalido: " + path;
  out.resize(static_cast<size_t>(end));
  f.seekg(0);
  if (!out.empty())
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()));
  if (!f) return "erro ao ler " + path;
  return "";
}

std::string write_atomic(const std::string &path, const std::vector<uint8_t> &bytes) {
  namespace fs = std::filesystem;
  const fs::path destination(path);
  std::error_code ec;
  if (!destination.parent_path().empty()) {
    fs::create_directories(destination.parent_path(), ec);
    if (ec) return "nao consegui criar a pasta " + destination.parent_path().string();
  }
  const std::string temporary = path + ".tmp";
  {
    std::ofstream f(temporary, std::ios::binary | std::ios::trunc);
    if (!f) return "nao consegui criar " + temporary;
    if (!bytes.empty())
      f.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    f.flush();
    if (!f) return "erro ao escrever " + temporary;
  }
  fs::rename(temporary, path, ec);
  if (ec) {
    fs::remove(temporary, ec);
    return "nao consegui substituir " + path;
  }
  return "";
}

uint32_t token_crc(const std::vector<uint32_t> &tokens) {
  std::vector<uint8_t> bytes;
  bytes.reserve(tokens.size() * 4);
  for (uint32_t id : tokens) {
    bytes.push_back(static_cast<uint8_t>(id));
    bytes.push_back(static_cast<uint8_t>(id >> 8));
    bytes.push_back(static_cast<uint8_t>(id >> 16));
    bytes.push_back(static_cast<uint8_t>(id >> 24));
  }
  return crc32(bytes.data(), bytes.size());
}

float scheduled_learning_rate(uint64_t step, const TrainingConfig &cfg) {
  if (cfg.warmup_steps > 0 && step <= cfg.warmup_steps)
    return cfg.learning_rate * static_cast<float>(step) /
           static_cast<float>(cfg.warmup_steps);
  const uint64_t decay_steps =
      cfg.steps > cfg.warmup_steps ? cfg.steps - cfg.warmup_steps : 1;
  const uint64_t elapsed = step > cfg.warmup_steps ? step - cfg.warmup_steps : 0;
  const double progress = std::min(1.0, static_cast<double>(elapsed) /
                                            static_cast<double>(decay_steps));
  const double cosine = 0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
  const double ratio = cfg.min_learning_rate_ratio +
                       (1.0 - cfg.min_learning_rate_ratio) * cosine;
  return static_cast<float>(cfg.learning_rate * ratio);
}

std::string save_checkpoint(const std::string &path, const Model &model,
                            const AdamW &optimizer, const std::mt19937 &rng,
                            uint32_t train_hash, uint32_t validation_hash) {
  std::string err;
  const std::vector<uint8_t> model_bytes = serialize(model, err);
  if (!err.empty()) return "checkpoint: " + err;
  const std::vector<uint8_t> optimizer_bytes = optimizer.save_state();
  std::ostringstream rng_stream;
  rng_stream << rng;
  const std::string rng_text = rng_stream.str();

  Writer w;
  w.raw(reinterpret_cast<const uint8_t *>("ZCKP"), 4);
  w.u32(1);
  w.u64(optimizer.step_count());
  w.u32(train_hash);
  w.u32(validation_hash);
  w.u64(static_cast<uint64_t>(model_bytes.size()));
  w.u64(static_cast<uint64_t>(optimizer_bytes.size()));
  w.u32(static_cast<uint32_t>(rng_text.size()));
  w.raw(model_bytes.data(), model_bytes.size());
  w.raw(optimizer_bytes.data(), optimizer_bytes.size());
  w.raw(reinterpret_cast<const uint8_t *>(rng_text.data()), rng_text.size());
  w.u32(crc32(w.bytes.data(), w.bytes.size()));
  return write_atomic(path, w.bytes);
}

std::string load_checkpoint(const std::string &path, Model &model,
                            AdamW &optimizer, std::mt19937 &rng,
                            uint32_t train_hash, uint32_t validation_hash) {
  std::vector<uint8_t> bytes;
  std::string err = read_bytes(path, bytes);
  if (!err.empty()) return err;
  if (bytes.size() < 4 + 4 + 8 + 4 + 4 + 8 + 8 + 4 + 4)
    return "checkpoint truncado";
  if (std::memcmp(bytes.data(), "ZCKP", 4) != 0)
    return "magic de checkpoint invalida";
  const uint8_t *tail = bytes.data() + bytes.size() - 4;
  const uint32_t stored =
      static_cast<uint32_t>(tail[0]) |
      (static_cast<uint32_t>(tail[1]) << 8) |
      (static_cast<uint32_t>(tail[2]) << 16) |
      (static_cast<uint32_t>(tail[3]) << 24);
  if (crc32(bytes.data(), bytes.size() - 4) != stored)
    return "checksum do checkpoint invalido";

  Reader r{bytes.data() + 4, tail};
  uint32_t version = 0, train_crc = 0, val_crc = 0, rng_size = 0;
  uint64_t step = 0, model_size = 0, optimizer_size = 0;
  if (!r.u32(version) || version != 1 || !r.u64(step) ||
      !r.u32(train_crc) || !r.u32(val_crc) ||
      !r.u64(model_size) || !r.u64(optimizer_size) || !r.u32(rng_size))
    return "cabecalho de checkpoint invalido";
  if (train_crc != train_hash || val_crc != validation_hash)
    return "o corpus nao corresponde ao checkpoint (CRC de dados diferente)";
  if (model_size > r.left() || optimizer_size > r.left() - model_size ||
      rng_size > r.left() - model_size - optimizer_size)
    return "tamanhos invalidos no checkpoint";
  if (model_size + optimizer_size + rng_size != r.left())
    return "dados inesperados no checkpoint";

  std::vector<uint8_t> model_bytes, optimizer_bytes, rng_bytes;
  if (!r.raw(model_bytes, static_cast<size_t>(model_size)) ||
      !r.raw(optimizer_bytes, static_cast<size_t>(optimizer_size)) ||
      !r.raw(rng_bytes, static_cast<size_t>(rng_size)))
    return "checkpoint truncado";
  Model restored;
  err = deserialize(model_bytes.data(), model_bytes.size(), restored);
  if (!err.empty()) return "modelo do checkpoint: " + err;
  if (restored.cfg.vocab_size != model.cfg.vocab_size ||
      restored.cfg.n_layers != model.cfg.n_layers ||
      restored.cfg.d_model != model.cfg.d_model ||
      restored.cfg.n_heads != model.cfg.n_heads ||
      restored.cfg.d_ff != model.cfg.d_ff ||
      restored.cfg.ctx_len != model.cfg.ctx_len ||
      restored.tok.tokens != model.tok.tokens)
    return "a configuracao do checkpoint difere da configuracao solicitada";
  err = optimizer.load_state(optimizer_bytes.data(), optimizer_bytes.size(), restored);
  if (!err.empty()) return err;
  if (optimizer.step_count() != step)
    return "passo do otimizador difere do passo do checkpoint";

  const std::string rng_text(rng_bytes.begin(), rng_bytes.end());
  std::istringstream rng_stream(rng_text);
  std::mt19937 restored_rng;
  if (!(rng_stream >> restored_rng))
    return "estado aleatorio invalido no checkpoint";
  model = std::move(restored);
  rng = restored_rng;
  return "";
}

void accumulate_gradients(Model &total, const Model &part, float scale) {
  if (total.tensors.size() != part.tensors.size())
    throw std::runtime_error("gradientes incompativeis durante acumulacao");
  for (size_t t = 0; t < total.tensors.size(); ++t) {
    if (total.tensors[t].data.size() != part.tensors[t].data.size())
      throw std::runtime_error("dimensao de gradiente incompativel");
    for (size_t i = 0; i < total.tensors[t].data.size(); ++i)
      total.tensors[t].data[i] += part.tensors[t].data[i] * scale;
  }
}

} // namespace

std::string load_token_ids(const std::string &path, uint32_t vocab_size,
                           std::vector<uint32_t> &out) {
  std::vector<uint8_t> bytes;
  std::string err = read_bytes(path, bytes);
  if (!err.empty()) return err;
  if (bytes.empty() || bytes.size() % 4 != 0)
    return "ficheiro de tokens vazio ou com tamanho nao divisivel por quatro: " + path;
  std::vector<uint32_t> tokens(bytes.size() / 4);
  for (size_t i = 0; i < tokens.size(); ++i) {
    const size_t p = i * 4;
    const uint32_t id = static_cast<uint32_t>(bytes[p]) |
                        (static_cast<uint32_t>(bytes[p + 1]) << 8) |
                        (static_cast<uint32_t>(bytes[p + 2]) << 16) |
                        (static_cast<uint32_t>(bytes[p + 3]) << 24);
    if (id >= vocab_size)
      return "ID de token fora do vocabulario na posicao " + std::to_string(i);
    tokens[i] = id;
  }
  out = std::move(tokens);
  return "";
}

float evaluate_loss(const Model &model, const std::vector<uint32_t> &tokens,
                    uint32_t sequence_length) {
  if (tokens.size() < 2 || sequence_length == 0)
    throw std::invalid_argument("dados insuficientes para avaliacao");
  const uint32_t length = std::min(sequence_length, model.cfg.ctx_len);
  double total_loss = 0.0;
  uint64_t total_tokens = 0;
  for (size_t start = 0; start + 1 < tokens.size();) {
    const size_t count = std::min(static_cast<size_t>(length), tokens.size() - start - 1);
    std::vector<uint32_t> input(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                                tokens.begin() + static_cast<std::ptrdiff_t>(start + count));
    std::vector<uint32_t> targets(
        tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
        tokens.begin() + static_cast<std::ptrdiff_t>(start + count + 1));
    const float loss = causal_loss(model, input, targets);
    total_loss += static_cast<double>(loss) * count;
    total_tokens += count;
    start += count;
  }
  return static_cast<float>(total_loss / total_tokens);
}

std::string train_model(Model &model,
                        const std::vector<uint32_t> &train_tokens,
                        const std::vector<uint32_t> &validation_tokens,
                        const TrainingConfig &cfg,
                        const std::string &model_path,
                        const std::string &checkpoint_path,
                        const std::string &resume_path,
                        TrainingResult &result) {
  try {
    if (cfg.steps == 0 || cfg.batch_size == 0 || cfg.batch_size > 1024 ||
        cfg.sequence_length == 0 || !(cfg.learning_rate > 0.0f) ||
        !(cfg.min_learning_rate_ratio > 0.0f) || cfg.min_learning_rate_ratio > 1.0f ||
        cfg.weight_decay < 0.0f || cfg.max_grad_norm < 0.0f ||
        !std::isfinite(cfg.learning_rate) || !std::isfinite(cfg.min_learning_rate_ratio) ||
        !std::isfinite(cfg.weight_decay) || !std::isfinite(cfg.max_grad_norm) ||
        cfg.warmup_steps >= cfg.steps)
      return "hiperparametros de treino invalidos (steps > warmup_steps, batch e learning rate positivos)";
    if (!validate_config(model.cfg).empty() ||
        model.tensors.size() != tensor_specs(model.cfg).size())
      return "modelo invalido ou incompleto";
    const uint32_t sequence_length = std::min(cfg.sequence_length, model.cfg.ctx_len);
    if (train_tokens.size() <= sequence_length || validation_tokens.size() < 2)
      return "corpus pequeno demais para a janela de treino";
    for (uint32_t id : train_tokens)
      if (id >= model.cfg.vocab_size) return "token de treino fora do vocabulario";
    for (uint32_t id : validation_tokens)
      if (id >= model.cfg.vocab_size) return "token de validacao fora do vocabulario";

    AdamWConfig adam_cfg;
    adam_cfg.learning_rate = cfg.learning_rate;
    adam_cfg.weight_decay = cfg.weight_decay;
    adam_cfg.max_grad_norm = cfg.max_grad_norm;
    AdamW optimizer(model, adam_cfg);
    std::mt19937 rng(cfg.seed);
    const uint32_t train_hash = token_crc(train_tokens);
    const uint32_t validation_hash = token_crc(validation_tokens);
    if (!resume_path.empty()) {
      const std::string err = load_checkpoint(resume_path, model, optimizer, rng,
                                              train_hash, validation_hash);
      if (!err.empty()) return "nao foi possivel retomar: " + err;
    }

    result.initial_step = optimizer.step_count();
    if (cfg.steps <= result.initial_step)
      return "--steps tem de ser maior que o passo guardado no checkpoint";
    result.initial_train_loss = evaluate_loss(model, train_tokens, sequence_length);
    const auto started = std::chrono::steady_clock::now();
    std::cout << "Inicio: step " << result.initial_step
              << ", train loss " << result.initial_train_loss
              << ", val loss "
              << evaluate_loss(model, validation_tokens, sequence_length) << "\n";

    std::uniform_int_distribution<size_t> choose_start(
        0, train_tokens.size() - static_cast<size_t>(sequence_length) - 1);
    for (uint64_t step = optimizer.step_count() + 1; step <= cfg.steps; ++step) {
      optimizer.set_learning_rate(scheduled_learning_rate(step, cfg));
      Model batch_gradients;
      double batch_loss = 0.0;
      for (uint32_t b = 0; b < cfg.batch_size; ++b) {
        const size_t start = choose_start(rng);
        std::vector<uint32_t> input(
            train_tokens.begin() + static_cast<std::ptrdiff_t>(start),
            train_tokens.begin() + static_cast<std::ptrdiff_t>(start + sequence_length));
        std::vector<uint32_t> targets(
            train_tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
            train_tokens.begin() + static_cast<std::ptrdiff_t>(start + sequence_length + 1));
        Model gradients;
        batch_loss += loss_and_backward(model, input, targets, gradients);
        if (b == 0) batch_gradients = std::move(gradients);
        else accumulate_gradients(batch_gradients, gradients, 1.0f);
      }
      if (cfg.batch_size > 1)
        for (Tensor &t : batch_gradients.tensors)
          for (float &v : t.data) v /= static_cast<float>(cfg.batch_size);
      optimizer.step(model, batch_gradients);

      const bool log_now = cfg.log_every > 0 &&
                           (step % cfg.log_every == 0 || step == cfg.steps);
      if (log_now) {
        const float train_loss = evaluate_loss(model, train_tokens, sequence_length);
        const float val_loss = evaluate_loss(model, validation_tokens, sequence_length);
        std::cout << "step " << step << "/" << cfg.steps
                  << " batch_loss " << (batch_loss / cfg.batch_size)
                  << " train_loss " << train_loss
                  << " val_loss " << val_loss
                  << " lr " << scheduled_learning_rate(step, cfg) << "\n";
      }
      if (!checkpoint_path.empty() && cfg.checkpoint_every > 0 &&
          step % cfg.checkpoint_every == 0) {
        const std::string err = save_checkpoint(checkpoint_path, model, optimizer,
                                                rng, train_hash, validation_hash);
        if (!err.empty()) return err;
      }
    }

    result.final_step = optimizer.step_count();
    result.final_train_loss = evaluate_loss(model, train_tokens, sequence_length);
    result.final_validation_loss =
        evaluate_loss(model, validation_tokens, sequence_length);
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    if (!model_path.empty()) {
      const std::string err = write_zika(model, model_path);
      if (!err.empty()) return err;
    }
    if (!checkpoint_path.empty()) {
      const std::string err = save_checkpoint(checkpoint_path, model, optimizer,
                                              rng, train_hash, validation_hash);
      if (!err.empty()) return err;
    }
    std::cout << "Fim: step " << result.final_step
              << ", train loss " << result.final_train_loss
              << ", val loss " << result.final_validation_loss
              << ", tempo " << result.elapsed_seconds << "s\n";
    return "";
  } catch (const std::exception &ex) {
    return std::string("falha no treino: ") + ex.what();
  }
}

} // namespace zika

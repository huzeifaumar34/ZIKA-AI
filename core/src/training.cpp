#include "zika/training.hpp"

#include "zika/optimizer.hpp"
#include "zika/transformer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
  void f32(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    u32(bits);
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
  bool f32(float &f) {
    uint32_t bits = 0;
    if (!u32(bits)) return false;
    std::memcpy(&f, &bits, sizeof(f));
    return true;
  }
  bool raw(std::vector<uint8_t> &out, size_t n) {
    if (left() < n) return false;
    out.assign(p, p + n);
    p += n;
    return true;
  }
};

// ZCKP v2 guarda o estado do scheduler. ZCKP v1 continua legivel, mas a
// retoma exige que o utilizador indique o horizonte original explicitamente.
struct SchedulerState {
  float base_lr = 0.0f;
  float min_lr = 0.0f;
  float segment_start_lr = 0.0f;
  float current_lr = 0.0f;
  uint64_t initial_warmup_steps = 0;
  uint64_t segment_start_step = 0;
  uint64_t segment_warmup_end_step = 0;
  uint64_t segment_end_step = 0;
  uint32_t batch_size = 1;
  uint32_t sequence_length = 1;
  float weight_decay = 0.0f;
  float max_grad_norm = 0.0f;
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

float schedule_learning_rate(uint64_t step, const SchedulerState &s) {
  if (step <= s.initial_warmup_steps && s.initial_warmup_steps > 0)
    return s.base_lr * static_cast<float>(step) /
           static_cast<float>(s.initial_warmup_steps);

  if (step <= s.segment_start_step) return s.segment_start_lr;

  if (s.segment_warmup_end_step > s.segment_start_step &&
      step <= s.segment_warmup_end_step) {
    const double progress =
        static_cast<double>(step - s.segment_start_step) /
        static_cast<double>(s.segment_warmup_end_step - s.segment_start_step);
    return static_cast<float>(s.segment_start_lr +
        (s.base_lr - s.segment_start_lr) * progress);
  }

  const uint64_t decay_start = s.segment_warmup_end_step;
  if (s.segment_end_step <= decay_start) return s.min_lr;
  const double progress = std::min(
      1.0, static_cast<double>(step - decay_start) /
               static_cast<double>(s.segment_end_step - decay_start));
  const double cosine = 0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
  const float decay_start_lr =
      s.segment_warmup_end_step > s.segment_start_step
          ? s.base_lr : s.segment_start_lr;
  return static_cast<float>(s.min_lr + (decay_start_lr - s.min_lr) * cosine);
}

SchedulerState fresh_scheduler(const TrainingConfig &cfg) {
  SchedulerState s;
  s.base_lr = cfg.learning_rate;
  s.min_lr = cfg.learning_rate * cfg.min_learning_rate_ratio;
  s.segment_start_lr = cfg.learning_rate;
  s.initial_warmup_steps = cfg.warmup_steps;
  s.segment_start_step = cfg.warmup_steps;
  s.segment_warmup_end_step = cfg.warmup_steps;
  s.segment_end_step = cfg.steps;
  s.batch_size = cfg.batch_size;
  s.sequence_length = cfg.sequence_length;
  s.weight_decay = cfg.weight_decay;
  s.max_grad_norm = cfg.max_grad_norm;
  return s;
}

void extend_scheduler(SchedulerState &s, uint64_t current_step,
                      uint64_t new_end_step, uint32_t resume_warmup_steps) {
  const float current_lr = schedule_learning_rate(current_step, s);
  const bool previous_schedule_finished = current_step >= s.segment_end_step;
  s.segment_start_step = current_step;
  s.segment_start_lr = current_lr;
  s.segment_end_step = new_end_step;
  s.segment_warmup_end_step = current_step;
  if (previous_schedule_finished && new_end_step - current_step > 1) {
    const uint64_t remaining = new_end_step - current_step;
    const uint64_t ramp = std::min<uint64_t>(
        resume_warmup_steps, std::max<uint64_t>(1, remaining / 10));
    s.segment_warmup_end_step = current_step + ramp;
  }
  s.current_lr = current_lr;
}

std::string save_checkpoint(const std::string &path, const Model &model,
                            const AdamW &optimizer, const std::mt19937 &rng,
                            uint32_t train_hash, uint32_t validation_hash,
                            const SchedulerState &scheduler) {
  std::string err;
  const std::vector<uint8_t> model_bytes = serialize(model, err);
  if (!err.empty()) return "checkpoint: " + err;
  const std::vector<uint8_t> optimizer_bytes = optimizer.save_state();
  std::ostringstream rng_stream;
  rng_stream << rng;
  const std::string rng_text = rng_stream.str();

  Writer w;
  w.raw(reinterpret_cast<const uint8_t *>("ZCKP"), 4);
  w.u32(2);
  w.u64(optimizer.step_count());
  w.u32(train_hash);
  w.u32(validation_hash);
  w.u64(static_cast<uint64_t>(model_bytes.size()));
  w.u64(static_cast<uint64_t>(optimizer_bytes.size()));
  w.u32(static_cast<uint32_t>(rng_text.size()));
  w.f32(scheduler.base_lr);
  w.f32(scheduler.min_lr);
  w.f32(scheduler.segment_start_lr);
  w.f32(scheduler.current_lr);
  w.f32(scheduler.weight_decay);
  w.f32(scheduler.max_grad_norm);
  w.u32(scheduler.batch_size);
  w.u32(scheduler.sequence_length);
  w.u64(scheduler.initial_warmup_steps);
  w.u64(scheduler.segment_start_step);
  w.u64(scheduler.segment_warmup_end_step);
  w.u64(scheduler.segment_end_step);
  w.raw(model_bytes.data(), model_bytes.size());
  w.raw(optimizer_bytes.data(), optimizer_bytes.size());
  w.raw(reinterpret_cast<const uint8_t *>(rng_text.data()), rng_text.size());
  w.u32(crc32(w.bytes.data(), w.bytes.size()));
  return write_atomic(path, w.bytes);
}

std::string load_checkpoint(const std::string &path, Model &model,
                            AdamW &optimizer, std::mt19937 &rng,
                            uint32_t train_hash, uint32_t validation_hash,
                            const TrainingConfig &cfg, SchedulerState &scheduler) {
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
  if (!r.u32(version) || (version != 1 && version != 2) || !r.u64(step) ||
      !r.u32(train_crc) || !r.u32(val_crc) ||
      !r.u64(model_size) || !r.u64(optimizer_size) || !r.u32(rng_size))
    return "cabecalho de checkpoint invalido";
  if (train_crc != train_hash || val_crc != validation_hash)
    return "o corpus nao corresponde ao checkpoint (CRC de dados diferente)";

  if (version == 1) {
    if (cfg.legacy_resume_total_steps == 0 ||
        cfg.legacy_resume_total_steps < step)
      return "checkpoint ZCKP v1: indique --resume-legacy-total-steps com o total original de passos";
    scheduler = fresh_scheduler(cfg);
    scheduler.initial_warmup_steps = std::max<uint64_t>(
        1, cfg.legacy_resume_total_steps / 10);
    scheduler.segment_start_step = scheduler.initial_warmup_steps;
    scheduler.segment_warmup_end_step = scheduler.initial_warmup_steps;
    scheduler.segment_end_step = cfg.legacy_resume_total_steps;
    scheduler.current_lr = schedule_learning_rate(step, scheduler);
  } else {
    if (!r.f32(scheduler.base_lr) || !r.f32(scheduler.min_lr) ||
        !r.f32(scheduler.segment_start_lr) || !r.f32(scheduler.current_lr) ||
        !r.f32(scheduler.weight_decay) || !r.f32(scheduler.max_grad_norm) ||
        !r.u32(scheduler.batch_size) || !r.u32(scheduler.sequence_length) ||
        !r.u64(scheduler.initial_warmup_steps) ||
        !r.u64(scheduler.segment_start_step) ||
        !r.u64(scheduler.segment_warmup_end_step) ||
        !r.u64(scheduler.segment_end_step))
      return "metadados do scheduler no checkpoint truncados";
    if (!std::isfinite(scheduler.base_lr) || !std::isfinite(scheduler.min_lr) ||
        !std::isfinite(scheduler.segment_start_lr) ||
        !std::isfinite(scheduler.current_lr) ||
        scheduler.base_lr <= 0.0f || scheduler.min_lr <= 0.0f ||
        scheduler.min_lr > scheduler.base_lr ||
        scheduler.segment_start_step > scheduler.segment_warmup_end_step ||
        scheduler.segment_warmup_end_step > scheduler.segment_end_step ||
        step > scheduler.segment_end_step)
      return "metadados do scheduler invalidos";
    if (std::abs(scheduler.base_lr - cfg.learning_rate) > 1.0e-9f ||
        std::abs(scheduler.weight_decay - cfg.weight_decay) > 1.0e-9f ||
        std::abs(scheduler.max_grad_norm - cfg.max_grad_norm) > 1.0e-9f ||
        std::abs(scheduler.min_lr - cfg.learning_rate * cfg.min_learning_rate_ratio) > 1.0e-9f ||
        scheduler.batch_size != cfg.batch_size ||
        scheduler.sequence_length != cfg.sequence_length)
      return "a configuracao de treino difere do checkpoint (batch, seq-len, LR, min-LR, weight decay ou grad clip)";
  }

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

bool finite_model(const Model &model) {
  for (const Tensor &tensor : model.tensors)
    for (float value : tensor.data)
      if (!std::isfinite(value)) return false;
  return true;
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
                    uint32_t sequence_length, uint64_t max_tokens) {
  if (tokens.size() < 2 || sequence_length == 0)
    throw std::invalid_argument("dados insuficientes para avaliacao");
  const size_t length = std::min<size_t>(sequence_length, model.cfg.ctx_len);
  const size_t prediction_tokens = tokens.size() - 1;
  const bool complete = max_tokens == 0 || max_tokens >= prediction_tokens;
  if (complete) {
    double total_loss = 0.0;
    uint64_t total_count = 0;
    for (size_t start = 0; start < prediction_tokens;) {
      const size_t count = std::min(length, prediction_tokens - start);
      std::vector<uint32_t> input(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                                  tokens.begin() + static_cast<std::ptrdiff_t>(start + count));
      std::vector<uint32_t> targets(tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                    tokens.begin() + static_cast<std::ptrdiff_t>(start + count + 1));
      total_loss += static_cast<double>(causal_loss(model, input, targets)) * count;
      total_count += count;
      start += count;
    }
    return static_cast<float>(total_loss / total_count);
  }

  const size_t budget = static_cast<size_t>(max_tokens);
  const size_t windows = std::max<size_t>(1, (budget + length - 1) / length);
  const size_t window_tokens = std::max<size_t>(1, budget / windows);
  const size_t max_start = prediction_tokens - window_tokens;
  double total_loss = 0.0;
  uint64_t total_count = 0;
  for (size_t i = 0; i < windows; ++i) {
    const size_t start = windows == 1 ? max_start / 2
        : (i * max_start) / (windows - 1);
    const size_t count = std::min(window_tokens, prediction_tokens - start);
    std::vector<uint32_t> input(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                                tokens.begin() + static_cast<std::ptrdiff_t>(start + count));
    std::vector<uint32_t> targets(tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                  tokens.begin() + static_cast<std::ptrdiff_t>(start + count + 1));
    total_loss += static_cast<double>(causal_loss(model, input, targets)) * count;
    total_count += count;
  }
  return static_cast<float>(total_loss / total_count);
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
        cfg.eval_tokens == 0)
      return "hiperparametros de treino invalidos";
    if (resume_path.empty() && cfg.warmup_steps >= cfg.steps)
      return "warmup_steps tem de ser inferior a steps num treino novo";
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
    SchedulerState scheduler = fresh_scheduler(cfg);

    if (!resume_path.empty()) {
      const std::string err = load_checkpoint(
          resume_path, model, optimizer, rng, train_hash, validation_hash, cfg, scheduler);
      if (!err.empty()) return "nao foi possivel retomar: " + err;
    }

    result.initial_step = optimizer.step_count();
    if (cfg.steps <= result.initial_step)
      return "--steps tem de ser maior que o passo guardado no checkpoint";
    if (cfg.steps < scheduler.segment_end_step)
      return "--steps nao pode ser inferior ao horizonte guardado no checkpoint";
    if (cfg.steps > scheduler.segment_end_step)
      extend_scheduler(scheduler, result.initial_step, cfg.steps, cfg.resume_warmup_steps);
    const uint64_t remaining_steps = cfg.steps - result.initial_step;
    const uint64_t run_end_step = (cfg.max_steps_this_run == 0 ||
                                   cfg.max_steps_this_run >= remaining_steps)
                                      ? cfg.steps
                                      : result.initial_step + cfg.max_steps_this_run;

    result.initial_learning_rate = schedule_learning_rate(result.initial_step, scheduler);
    result.initial_train_loss = evaluate_loss(
        model, train_tokens, sequence_length, cfg.eval_tokens);
    result.initial_validation_loss = evaluate_loss(
        model, validation_tokens, sequence_length, cfg.eval_tokens);
    const auto started = std::chrono::steady_clock::now();
    std::cout << "Inicio: step " << result.initial_step
              << ", train_loss_sample " << result.initial_train_loss
              << ", val_loss_sample " << result.initial_validation_loss
              << ", lr " << result.initial_learning_rate << "\n";

    std::uniform_int_distribution<size_t> choose_start(
        0, train_tokens.size() - static_cast<size_t>(sequence_length) - 1);
    for (uint64_t step = optimizer.step_count() + 1; step <= run_end_step; ++step) {
      const float learning_rate = schedule_learning_rate(step, scheduler);
      optimizer.set_learning_rate(learning_rate);
      scheduler.current_lr = learning_rate;
      if (step == result.initial_step + 1)
        result.first_learning_rate = learning_rate;

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
      if (!finite_model(model))
        return "treino interrompido: pesos NaN ou infinitos detetados";

      if (cfg.log_every > 0 &&
          (step % cfg.log_every == 0 || step == run_end_step)) {
        std::cout << "step " << step << "/" << cfg.steps
                  << " batch_loss " << (batch_loss / cfg.batch_size)
                  << " lr " << learning_rate << "\n";
      }

      if (cfg.eval_every > 0 &&
          (step % cfg.eval_every == 0 || step == run_end_step)) {
        const float train_sample = evaluate_loss(
            model, train_tokens, sequence_length, cfg.eval_tokens);
        const float val_sample = evaluate_loss(
            model, validation_tokens, sequence_length, cfg.eval_tokens);
        ++result.sampled_evaluations;
        std::cout << "eval step " << step
                  << " train_sample_loss " << train_sample
                  << " val_sample_loss " << val_sample;
        if (cfg.full_eval_every > 0 &&
            (step % cfg.full_eval_every == 0 || step == run_end_step)) {
          const float full_val = evaluate_loss(
              model, validation_tokens, sequence_length, 0);
          ++result.full_evaluations;
          std::cout << " val_full_loss " << full_val;
        }
        std::cout << "\n";
      } else if (cfg.full_eval_every > 0 &&
                 (step % cfg.full_eval_every == 0 || step == run_end_step)) {
        const float full_val = evaluate_loss(
            model, validation_tokens, sequence_length, 0);
        ++result.full_evaluations;
        std::cout << "eval step " << step << " val_full_loss " << full_val << "\n";
      }

      if (!checkpoint_path.empty() && cfg.checkpoint_every > 0 &&
          step % cfg.checkpoint_every == 0) {
        const std::string err = save_checkpoint(
            checkpoint_path, model, optimizer, rng, train_hash, validation_hash, scheduler);
        if (!err.empty()) return err;
      }
    }

    result.final_step = optimizer.step_count();
    result.final_learning_rate = scheduler.current_lr;
    result.final_train_loss = evaluate_loss(model, train_tokens, sequence_length, 0);
    result.final_validation_loss = evaluate_loss(model, validation_tokens, sequence_length, 0);
    ++result.full_evaluations;
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    if (!finite_model(model)) return "treino terminou com pesos NaN ou infinitos";
    if (!model_path.empty()) {
      const std::string err = write_zika(model, model_path);
      if (!err.empty()) return err;
    }
    if (!checkpoint_path.empty()) {
      const std::string err = save_checkpoint(
          checkpoint_path, model, optimizer, rng, train_hash, validation_hash, scheduler);
      if (!err.empty()) return err;
    }
    std::cout << "Fim: step " << result.final_step
              << ", train_loss_full " << result.final_train_loss
              << ", val_loss_full " << result.final_validation_loss
              << ", sampled_evals " << result.sampled_evaluations
              << ", full_evals " << result.full_evaluations
              << ", tempo " << result.elapsed_seconds << "s\n";
    return "";
  } catch (const std::exception &ex) {
    return std::string("falha no treino: ") + ex.what();
  }
}

} // namespace zika

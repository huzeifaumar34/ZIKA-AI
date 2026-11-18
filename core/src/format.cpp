#include "zika/format.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace zika {

uint64_t Tensor::numel() const {
  uint64_t n = 1;
  for (uint32_t d : shape)
    n *= d;
  return n;
}

namespace {

constexpr char kMagic[4] = {'Z', 'I', 'K', 'A'};
constexpr uint32_t kMaxTokenLen = 64;
constexpr uint32_t kMaxMerges = 1u << 20;
constexpr uint32_t kMaxTensors = 100000;
constexpr uint32_t kMaxNameLen = 256;
constexpr uint32_t kMaxDims = 4;
constexpr uint64_t kMaxFileSize = 1ull << 32; // 4 GiB

// ---------- escrita ----------

struct Writer {
  std::vector<uint8_t> buf;

  void u32(uint32_t v) {
    for (int i = 0; i < 4; i++)
      buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void bytes(const void *p, size_t n) {
    const auto *b = static_cast<const uint8_t *>(p);
    buf.insert(buf.end(), b, b + n);
  }
  void f32(float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    u32(v);
  }
};

// ---------- leitura (com verificação de limites) ----------

struct Reader {
  const uint8_t *p;
  const uint8_t *end;

  size_t left() const { return static_cast<size_t>(end - p); }

  bool u32(uint32_t &v) {
    if (left() < 4)
      return false;
    v = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return true;
  }
  bool bytes(void *dst, size_t n) {
    if (left() < n)
      return false;
    std::memcpy(dst, p, n);
    p += n;
    return true;
  }
  bool f32(float &f) {
    uint32_t v;
    if (!u32(v))
      return false;
    std::memcpy(&f, &v, 4);
    return true;
  }
};

const char *kTruncated = "ficheiro truncado ou corrompido";

std::string validate_tokenizer_data(const Tokenizer &tokenizer) {
  if (tokenizer.kind == TokenizerKind::Char) {
    if (!tokenizer.merges.empty())
      return "tokenizer char nao pode conter regras BPE";
    return "";
  }
  if (tokenizer.kind != TokenizerKind::BPE)
    return "tipo de tokenizer desconhecido";

  constexpr size_t base_size = 3 + 256;
  if (tokenizer.tokens.size() < base_size)
    return "vocabulario BPE nao contem os 256 tokens de byte";
  const std::string special[] = {"<|user|>", "<|ia|>", "<|fim|>"};
  for (size_t i = 0; i < 3; ++i)
    if (tokenizer.tokens[i] != special[i])
      return "os tres tokens especiais BPE estao fora de ordem";
  for (size_t value = 0; value < 256; ++value) {
    const std::string expected(1, static_cast<char>(value));
    if (tokenizer.tokens[value + 3] != expected)
      return "tokens base BPE nao correspondem a ordem byte-v0";
  }
  if (tokenizer.merges.size() != tokenizer.tokens.size() - base_size)
    return "numero de merges BPE difere do tamanho do vocabulario";
  for (size_t i = 0; i < tokenizer.merges.size(); ++i) {
    const uint32_t left = tokenizer.merges[i].first;
    const uint32_t right = tokenizer.merges[i].second;
    const size_t new_id = base_size + i;
    if (left < 3 || right < 3 || left >= new_id || right >= new_id)
      return "merge BPE invalido na posicao " + std::to_string(i);
    const std::string expected = tokenizer.tokens[left] + tokenizer.tokens[right];
    if (expected.size() > kMaxTokenLen || tokenizer.tokens[new_id] != expected)
      return "token BPE nao corresponde ao merge " + std::to_string(i);
  }
  return "";
}

} // namespace

uint32_t crc32(const uint8_t *data, size_t size) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++)
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      t[i] = c;
    }
    return t;
  }();
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < size; i++)
    c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

std::string validate_config(const Config &c) {
  if (c.vocab_size == 0 || c.vocab_size > (1u << 20))
    return "vocab_size tem de estar entre 1 e 1048576";
  if (c.n_layers == 0 || c.n_layers > 128)
    return "n_layers tem de estar entre 1 e 128";
  if (c.d_model == 0 || c.d_model > 8192)
    return "d_model tem de estar entre 1 e 8192";
  if (c.n_heads == 0 || c.n_heads > c.d_model)
    return "n_heads tem de estar entre 1 e d_model";
  if (c.d_model % c.n_heads != 0)
    return "d_model tem de ser divisivel por n_heads";
  if (c.d_ff == 0 || c.d_ff > 65536)
    return "d_ff tem de estar entre 1 e 65536";
  if (c.ctx_len == 0 || c.ctx_len > 65536)
    return "ctx_len tem de estar entre 1 e 65536";
  return "";
}

std::vector<TensorSpec> tensor_specs(const Config &c) {
  const uint32_t d = c.d_model;
  const uint32_t f = c.d_ff;
  std::vector<TensorSpec> s;
  s.push_back({"tok_emb", {c.vocab_size, d}});
  s.push_back({"pos_emb", {c.ctx_len, d}});
  for (uint32_t l = 0; l < c.n_layers; l++) {
    const std::string p = "layer" + std::to_string(l) + ".";
    s.push_back({p + "ln1.g", {d}});
    s.push_back({p + "ln1.b", {d}});
    s.push_back({p + "attn.wqkv", {d, 3 * d}});
    s.push_back({p + "attn.bqkv", {3 * d}});
    s.push_back({p + "attn.wo", {d, d}});
    s.push_back({p + "attn.bo", {d}});
    s.push_back({p + "ln2.g", {d}});
    s.push_back({p + "ln2.b", {d}});
    s.push_back({p + "mlp.w1", {d, f}});
    s.push_back({p + "mlp.b1", {f}});
    s.push_back({p + "mlp.w2", {f, d}});
    s.push_back({p + "mlp.b2", {d}});
  }
  s.push_back({"lnf.g", {d}});
  s.push_back({"lnf.b", {d}});
  return s;
}

uint64_t param_count(const Config &c) {
  uint64_t total = 0;
  for (const TensorSpec &s : tensor_specs(c)) {
    uint64_t n = 1;
    for (uint32_t d : s.shape)
      n *= d;
    total += n;
  }
  return total;
}

std::vector<std::string> default_char_vocab() {
  std::vector<std::string> v = {"<|user|>", "<|ia|>", "<|fim|>", "\n"};
  for (int c = 32; c <= 126; c++)
    v.push_back(std::string(1, static_cast<char>(c)));
  const char *accents[] = {"á", "à", "â", "ã", "é", "ê", "í", "ó", "ô",
                           "õ", "ú", "ü", "ç", "Á", "À", "Â", "Ã", "É",
                           "Ê", "Í", "Ó", "Ô", "Õ", "Ú", "Ç"};
  for (const char *a : accents)
    v.push_back(a);
  return v;
}

Model make_empty_model(const Config &cfg, const Tokenizer &tok) {
  Model m;
  m.cfg = cfg;
  m.tok = tok;
  for (const TensorSpec &s : tensor_specs(cfg)) {
    Tensor t;
    t.name = s.name;
    t.shape = s.shape;
    const bool is_gain =
        t.name.size() >= 2 && t.name.compare(t.name.size() - 2, 2, ".g") == 0;
    t.data.assign(t.numel(), is_gain ? 1.0f : 0.0f);
    m.tensors.push_back(std::move(t));
  }
  return m;
}

std::vector<uint8_t> serialize(const Model &m, std::string &err) {
  err.clear();
  if (m.dtype != DType::F32) {
    err = "só o tipo de dados F32 está implementado";
    return {};
  }
  err = validate_config(m.cfg);
  if (!err.empty())
    return {};
  if (m.tok.tokens.size() != m.cfg.vocab_size) {
    err = "o numero de tokens do vocabulario difere de vocab_size";
    return {};
  }
  err = validate_tokenizer_data(m.tok);
  if (!err.empty()) return {};
  for (const std::string &t : m.tok.tokens) {
    if (t.empty() || t.size() > kMaxTokenLen) {
      err = "token do vocabulario com tamanho invalido";
      return {};
    }
  }
  if (m.tok.merges.size() > kMaxMerges) {
    err = "demasiadas regras BPE";
    return {};
  }
  const std::vector<TensorSpec> specs = tensor_specs(m.cfg);
  if (m.tensors.size() != specs.size()) {
    err = "o numero de tensores nao corresponde a configuracao";
    return {};
  }
  for (size_t i = 0; i < specs.size(); i++) {
    const Tensor &t = m.tensors[i];
    if (t.name != specs[i].name || t.shape != specs[i].shape ||
        t.data.size() != t.numel()) {
      err = "tensor inesperado na posicao " + std::to_string(i) + " (esperado " +
            specs[i].name + ")";
      return {};
    }
  }

  Writer w;
  uint64_t payload = 0;
  for (const Tensor &t : m.tensors)
    payload += t.data.size() * 4;
  w.buf.reserve(static_cast<size_t>(payload) + 4096);

  w.bytes(kMagic, 4);
  w.u32(kFormatVersion);
  w.u32(static_cast<uint32_t>(m.dtype));
  w.u32(m.cfg.vocab_size);
  w.u32(m.cfg.n_layers);
  w.u32(m.cfg.d_model);
  w.u32(m.cfg.n_heads);
  w.u32(m.cfg.d_ff);
  w.u32(m.cfg.ctx_len);
  w.u32(static_cast<uint32_t>(m.tok.kind));
  w.u32(static_cast<uint32_t>(m.tok.tokens.size()));
  for (const std::string &t : m.tok.tokens) {
    w.u32(static_cast<uint32_t>(t.size()));
    w.bytes(t.data(), t.size());
  }
  w.u32(static_cast<uint32_t>(m.tok.merges.size()));
  for (const auto &mg : m.tok.merges) {
    w.u32(mg.first);
    w.u32(mg.second);
  }
  w.u32(static_cast<uint32_t>(m.tensors.size()));
  for (const Tensor &t : m.tensors) {
    w.u32(static_cast<uint32_t>(t.name.size()));
    w.bytes(t.name.data(), t.name.size());
    w.u32(static_cast<uint32_t>(t.shape.size()));
    for (uint32_t d : t.shape)
      w.u32(d);
    for (float f : t.data)
      w.f32(f);
  }
  w.u32(crc32(w.buf.data(), w.buf.size()));
  return std::move(w.buf);
}

std::string deserialize(const uint8_t *data, size_t size, Model &out) {
  if (size < 12)
    return kTruncated;
  if (std::memcmp(data, kMagic, 4) != 0)
    return "magia invalida: nao e um ficheiro .zika";

  Reader r{data + 4, data + size};
  uint32_t version = 0;
  r.u32(version);
  if (version != kFormatVersion)
    return "versao do formato nao suportada: " + std::to_string(version);

  if (size < 8)
    return kTruncated;
  const uint8_t *tail = data + size - 4;
  const uint32_t stored = static_cast<uint32_t>(tail[0]) |
                          (static_cast<uint32_t>(tail[1]) << 8) |
                          (static_cast<uint32_t>(tail[2]) << 16) |
                          (static_cast<uint32_t>(tail[3]) << 24);
  if (crc32(data, size - 4) != stored)
    return "checksum invalido: ficheiro corrompido ou incompleto";
  r.end = tail; // o corpo termina antes do checksum

  Model m;
  uint32_t v = 0;
  if (!r.u32(v))
    return kTruncated;
  if (v != static_cast<uint32_t>(DType::F32))
    return "tipo de dados nao suportado: " + std::to_string(v);
  m.dtype = DType::F32;

  uint32_t* fields[] = {&m.cfg.vocab_size, &m.cfg.n_layers, &m.cfg.d_model,
                        &m.cfg.n_heads,    &m.cfg.d_ff,     &m.cfg.ctx_len};
  for (uint32_t *f : fields) {
    if (!r.u32(*f))
      return kTruncated;
  }
  std::string err = validate_config(m.cfg);
  if (!err.empty())
    return "configuracao invalida: " + err;

  if (!r.u32(v))
    return kTruncated;
  if (v > static_cast<uint32_t>(TokenizerKind::BPE))
    return "tipo de tokenizer desconhecido: " + std::to_string(v);
  m.tok.kind = static_cast<TokenizerKind>(v);

  uint32_t n_tokens = 0;
  if (!r.u32(n_tokens))
    return kTruncated;
  if (n_tokens != m.cfg.vocab_size)
    return "o numero de tokens difere de vocab_size";
  m.tok.tokens.reserve(n_tokens);
  for (uint32_t i = 0; i < n_tokens; i++) {
    uint32_t len = 0;
    if (!r.u32(len))
      return kTruncated;
    if (len == 0 || len > kMaxTokenLen)
      return "token com tamanho invalido";
    std::string s(len, '\0');
    if (!r.bytes(&s[0], len))
      return kTruncated;
    m.tok.tokens.push_back(std::move(s));
  }

  uint32_t n_merges = 0;
  if (!r.u32(n_merges))
    return kTruncated;
  if (n_merges > kMaxMerges || static_cast<uint64_t>(n_merges) * 8 > r.left())
    return "numero de regras BPE invalido";
  m.tok.merges.reserve(n_merges);
  for (uint32_t i = 0; i < n_merges; i++) {
    uint32_t a = 0, b = 0;
    if (!r.u32(a) || !r.u32(b))
      return kTruncated;
    m.tok.merges.emplace_back(a, b);
  }
  err = validate_tokenizer_data(m.tok);
  if (!err.empty()) return err;

  uint32_t n_tensors = 0;
  if (!r.u32(n_tensors))
    return kTruncated;
  const std::vector<TensorSpec> specs = tensor_specs(m.cfg);
  if (n_tensors > kMaxTensors || n_tensors != specs.size())
    return "o numero de tensores nao corresponde a configuracao";

  m.tensors.reserve(n_tensors);
  for (uint32_t i = 0; i < n_tensors; i++) {
    Tensor t;
    uint32_t name_len = 0;
    if (!r.u32(name_len))
      return kTruncated;
    if (name_len == 0 || name_len > kMaxNameLen)
      return "nome de tensor invalido";
    t.name.assign(name_len, '\0');
    if (!r.bytes(&t.name[0], name_len))
      return kTruncated;

    uint32_t ndim = 0;
    if (!r.u32(ndim))
      return kTruncated;
    if (ndim == 0 || ndim > kMaxDims)
      return "numero de dimensoes invalido";
    t.shape.resize(ndim);
    for (uint32_t k = 0; k < ndim; k++) {
      if (!r.u32(t.shape[k]))
        return kTruncated;
    }
    if (t.name != specs[i].name || t.shape != specs[i].shape)
      return "tensor inesperado na posicao " + std::to_string(i) +
             " (esperado " + specs[i].name + ")";

    const uint64_t n = t.numel();
    if (n * 4 > r.left())
      return kTruncated;
    t.data.resize(static_cast<size_t>(n));
    for (uint64_t k = 0; k < n; k++)
      r.f32(t.data[static_cast<size_t>(k)]);
    m.tensors.push_back(std::move(t));
  }
  if (r.left() != 0)
    return "dados a mais no fim do ficheiro";

  out = std::move(m);
  return "";
}

std::string write_zika(const Model &model, const std::string &path) {
  std::string err;
  const std::vector<uint8_t> bytes = serialize(model, err);
  if (!err.empty())
    return err;

  namespace fs = std::filesystem;
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f)
      return "nao consegui criar " + tmp;
    f.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    f.flush();
    if (!f)
      return "erro ao escrever " + tmp;
  }
  std::error_code ec;
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return "nao consegui gravar " + path;
  }
  return "";
}

std::string read_zika(const std::string &path, Model &out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    return "nao consegui abrir " + path;
  const std::streamoff end = f.tellg();
  if (end < 0 || static_cast<uint64_t>(end) > kMaxFileSize)
    return "tamanho de ficheiro invalido";
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(bytes.data()),
         static_cast<std::streamsize>(bytes.size()));
  if (!f)
    return "erro ao ler " + path;
  return deserialize(bytes.data(), bytes.size(), out);
}


std::string read_tokenizer(const std::string &path, Tokenizer &out) {
  constexpr uint64_t kMaxTokenizerFile = 1ull << 30;
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) return "nao consegui abrir " + path;
  const std::streamoff end = file.tellg();
  if (end < 20 || static_cast<uint64_t>(end) > kMaxTokenizerFile)
    return "tamanho de ficheiro ZTKZ invalido";
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!file) return "erro ao ler " + path;
  if (std::memcmp(bytes.data(), "ZTKZ", 4) != 0)
    return "magia invalida: nao e um tokenizer ZTKZ";

  const uint8_t *tail = bytes.data() + bytes.size() - 4;
  const uint32_t stored =
      static_cast<uint32_t>(tail[0]) |
      (static_cast<uint32_t>(tail[1]) << 8) |
      (static_cast<uint32_t>(tail[2]) << 16) |
      (static_cast<uint32_t>(tail[3]) << 24);
  if (crc32(bytes.data(), bytes.size() - 4) != stored)
    return "checksum invalido no tokenizer ZTKZ";

  Reader reader{bytes.data() + 4, tail};
  uint32_t version = 0, kind = 0, token_count = 0, merge_count = 0;
  if (!reader.u32(version) || !reader.u32(kind) ||
      !reader.u32(token_count) || !reader.u32(merge_count))
    return kTruncated;
  if (version != 1) return "versao ZTKZ nao suportada";
  if (kind != static_cast<uint32_t>(TokenizerKind::BPE))
    return "ZTKZ deve conter tokenizer BPE";
  if (token_count < 259 || token_count > (1u << 20) ||
      merge_count > (1u << 20) || merge_count != token_count - 259)
    return "contagens de tokens/merges ZTKZ invalidas";

  Tokenizer parsed;
  parsed.kind = TokenizerKind::BPE;
  parsed.tokens.reserve(token_count);
  for (uint32_t i = 0; i < token_count; ++i) {
    uint32_t length = 0;
    if (!reader.u32(length)) return kTruncated;
    if (length == 0 || length > kMaxTokenLen || length > reader.left())
      return "comprimento de token ZTKZ invalido";
    std::string token(length, '\0');
    if (!reader.bytes(&token[0], length)) return kTruncated;
    parsed.tokens.push_back(std::move(token));
  }
  parsed.merges.reserve(merge_count);
  for (uint32_t i = 0; i < merge_count; ++i) {
    uint32_t left = 0, right = 0;
    if (!reader.u32(left) || !reader.u32(right)) return kTruncated;
    parsed.merges.emplace_back(left, right);
  }
  if (reader.left() != 0) return "dados a mais no tokenizer ZTKZ";
  const std::string error = validate_tokenizer_data(parsed);
  if (!error.empty()) return error;
  out = std::move(parsed);
  return "";
}

} // namespace zika

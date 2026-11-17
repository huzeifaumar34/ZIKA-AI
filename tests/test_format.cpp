// Testes da fase 1: formato .zika, validação de configuração e leitor de JSON.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "zika/format.hpp"
#include "zika/spec.hpp"

using namespace zika;

static int g_checks = 0;
static int g_fail = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      ++g_fail;                                                                \
      std::fprintf(stderr, "FALHOU %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                          \
  } while (0)

// Verifica que a mensagem de erro contém um pedaço de texto.
#define CHECK_ERR(err, sub)                                                    \
  do {                                                                         \
    ++g_checks;                                                                \
    if ((err).find(sub) == std::string::npos) {                                \
      ++g_fail;                                                                \
      std::fprintf(stderr, "FALHOU %s:%d: erro \"%s\" nao contem \"%s\"\n",    \
                   __FILE__, __LINE__, (err).c_str(), sub);                    \
    }                                                                          \
  } while (0)

static Config tiny_config() {
  Config c;
  c.vocab_size = static_cast<uint32_t>(default_char_vocab().size());
  c.n_layers = 2;
  c.d_model = 16;
  c.n_heads = 2;
  c.d_ff = 32;
  c.ctx_len = 8;
  return c;
}

static Tokenizer char_tokenizer() {
  Tokenizer t;
  t.kind = TokenizerKind::Char;
  t.tokens = default_char_vocab();
  return t;
}

// Preenche com padrões de bits pseudo-aleatórios (inclui NaN, infinitos e -0).
static Model random_model(const Config &cfg) {
  Model m = make_empty_model(cfg, char_tokenizer());
  uint32_t s = 2463534242u;
  for (Tensor &t : m.tensors) {
    for (float &f : t.data) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      std::memcpy(&f, &s, 4);
    }
  }
  return m;
}

static bool same_bits(const Model &a, const Model &b) {
  if (std::memcmp(&a.cfg, &b.cfg, sizeof(Config)) != 0)
    return false;
  if (a.dtype != b.dtype || a.tok.kind != b.tok.kind ||
      a.tok.tokens != b.tok.tokens || a.tok.merges != b.tok.merges)
    return false;
  if (a.tensors.size() != b.tensors.size())
    return false;
  for (size_t i = 0; i < a.tensors.size(); i++) {
    const Tensor &x = a.tensors[i];
    const Tensor &y = b.tensors[i];
    if (x.name != y.name || x.shape != y.shape || x.data.size() != y.data.size())
      return false;
    if (std::memcmp(x.data.data(), y.data.data(), x.data.size() * 4) != 0)
      return false;
  }
  return true;
}

// Recalcula o checksum no fim do buffer (para testar erros que passam o CRC).
static void fix_crc(std::vector<uint8_t> &b) {
  const uint32_t c = crc32(b.data(), b.size() - 4);
  for (int i = 0; i < 4; i++)
    b[b.size() - 4 + i] = static_cast<uint8_t>(c >> (8 * i));
}

static void set_u32(std::vector<uint8_t> &b, size_t off, uint32_t v) {
  for (int i = 0; i < 4; i++)
    b[off + i] = static_cast<uint8_t>(v >> (8 * i));
}

static void test_crc32() {
  const char *s = "123456789";
  CHECK(crc32(reinterpret_cast<const uint8_t *>(s), 9) == 0xCBF43926u);
  CHECK(crc32(nullptr, 0) == 0u);
}

static void test_vocab_and_params() {
  CHECK(default_char_vocab().size() == 124);

  Config nano;
  nano.vocab_size = 124;
  nano.n_layers = 3;
  nano.d_model = 128;
  nano.n_heads = 4;
  nano.d_ff = 512;
  nano.ctx_len = 256;
  CHECK(validate_config(nano).empty());
  CHECK(param_count(nano) == 643712);

  Config mini = nano;
  mini.n_layers = 4;
  mini.d_model = 256;
  mini.n_heads = 8;
  mini.d_ff = 1024;
  CHECK(validate_config(mini).empty());
  CHECK(param_count(mini) == 3256832);

  // A soma dos tensores tem de bater certo com param_count.
  const Model m = make_empty_model(tiny_config(), char_tokenizer());
  uint64_t total = 0;
  for (const Tensor &t : m.tensors)
    total += t.numel();
  CHECK(total == param_count(tiny_config()));
}

static void test_validate_config() {
  Config c = tiny_config();
  CHECK(validate_config(c).empty());

  Config bad = c;
  bad.d_model = 17; // nao divisivel por n_heads
  CHECK_ERR(validate_config(bad), "divisivel");

  for (int field = 0; field < 6; field++) {
    Config z = c;
    uint32_t *fs[] = {&z.vocab_size, &z.n_layers, &z.d_model,
                      &z.n_heads,    &z.d_ff,     &z.ctx_len};
    *fs[field] = 0;
    CHECK(!validate_config(z).empty());
  }
}

static void test_roundtrip_memory() {
  const Model m = random_model(tiny_config());
  std::string err;
  const std::vector<uint8_t> bytes = serialize(m, err);
  CHECK(err.empty());
  CHECK(bytes.size() > 100);

  // Cabeçalho fixo: "ZIKA", versão 1, dtype F32 (tudo little-endian).
  const uint8_t head[12] = {'Z', 'I', 'K', 'A', 1, 0, 0, 0, 0, 0, 0, 0};
  CHECK(std::memcmp(bytes.data(), head, 12) == 0);

  Model back;
  err = deserialize(bytes.data(), bytes.size(), back);
  CHECK(err.empty());
  CHECK(same_bits(m, back));

  // Serializar duas vezes dá exatamente os mesmos bytes.
  std::string err2;
  CHECK(serialize(back, err2) == bytes);
}

static void test_roundtrip_file() {
  const Model m = random_model(tiny_config());
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "zika_test_format";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "modelo.zika").string();

  CHECK(write_zika(m, path).empty());
  CHECK(!std::filesystem::exists(path + ".tmp"));

  Model back;
  CHECK(read_zika(path, back).empty());
  CHECK(same_bits(m, back));

  Model none;
  CHECK(!read_zika((dir / "nao_existe.zika").string(), none).empty());
  std::filesystem::remove_all(dir);
}

static void test_empty_model() {
  const Model m = make_empty_model(tiny_config(), char_tokenizer());
  std::string err;
  const std::vector<uint8_t> bytes = serialize(m, err);
  CHECK(err.empty());
  Model back;
  CHECK(deserialize(bytes.data(), bytes.size(), back).empty());
  CHECK(back.tensors[0].name == "tok_emb");
  CHECK(back.tensors[0].data[0] == 0.0f);
  bool found_gain = false;
  for (const Tensor &t : back.tensors) {
    if (t.name == "lnf.g") {
      found_gain = true;
      CHECK(t.data[0] == 1.0f);
    }
  }
  CHECK(found_gain);
}

static void test_bad_files() {
  const Model m = random_model(tiny_config());
  std::string err;
  const std::vector<uint8_t> good = serialize(m, err);
  Model out;

  { // magia errada
    std::vector<uint8_t> b = good;
    b[0] = 'X';
    CHECK_ERR(deserialize(b.data(), b.size(), out), "magia");
  }
  { // versão desconhecida
    std::vector<uint8_t> b = good;
    b[4] = 99;
    CHECK_ERR(deserialize(b.data(), b.size(), out), "vers");
  }
  { // um byte alterado no meio dos dados
    std::vector<uint8_t> b = good;
    b[b.size() / 2] ^= 0x01;
    CHECK_ERR(deserialize(b.data(), b.size(), out), "checksum");
  }
  { // truncado
    std::vector<uint8_t> b(good.begin(), good.end() - 10);
    CHECK(!deserialize(b.data(), b.size(), out).empty());
  }
  { // lixo no fim
    std::vector<uint8_t> b = good;
    b.push_back(0);
    CHECK(!deserialize(b.data(), b.size(), out).empty());
  }
  { // ficheiros minúsculos e vazios
    CHECK(!deserialize(good.data(), 3, out).empty());
    CHECK(!deserialize(good.data(), 11, out).empty());
    CHECK(!deserialize(nullptr, 0, out).empty());
  }
  // Os casos seguintes têm o CRC recalculado, por isso só o leitor os apanha.
  { // d_model passa a nao ser divisivel por n_heads (offset 20)
    std::vector<uint8_t> b = good;
    set_u32(b, 20, 17);
    fix_crc(b);
    CHECK_ERR(deserialize(b.data(), b.size(), out), "divisivel");
  }
  { // vocab_size diferente do numero de tokens (offset 12)
    std::vector<uint8_t> b = good;
    set_u32(b, 12, 5);
    fix_crc(b);
    CHECK(!deserialize(b.data(), b.size(), out).empty());
  }
  { // dtype nao suportado (offset 8)
    std::vector<uint8_t> b = good;
    set_u32(b, 8, 2);
    fix_crc(b);
    CHECK_ERR(deserialize(b.data(), b.size(), out), "tipo de dados");
  }
  { // corpo cortado mas com CRC valido
    std::vector<uint8_t> b(good.begin(), good.end() - 104);
    b.insert(b.end(), 4, 0);
    fix_crc(b);
    CHECK(!deserialize(b.data(), b.size(), out).empty());
  }
}

static void test_serialize_rejects_bad_models() {
  std::string err;
  { // falta um tensor
    Model m = make_empty_model(tiny_config(), char_tokenizer());
    m.tensors.pop_back();
    serialize(m, err);
    CHECK(!err.empty());
  }
  { // dados com tamanho errado
    Model m = make_empty_model(tiny_config(), char_tokenizer());
    m.tensors[0].data.pop_back();
    serialize(m, err);
    CHECK(!err.empty());
  }
  { // vocabulário com número de tokens errado
    Model m = make_empty_model(tiny_config(), char_tokenizer());
    m.tok.tokens.pop_back();
    serialize(m, err);
    CHECK(!err.empty());
  }
  { // tipo de dados ainda não implementado
    Model m = make_empty_model(tiny_config(), char_tokenizer());
    m.dtype = DType::I8;
    serialize(m, err);
    CHECK(!err.empty());
  }
}

static void test_spec_json() {
  ModelSpec s;
  const std::string ok = R"({
    "name": "teste", "tokenizer": "char",
    "n_layers": 2, "d_model": 64, "n_heads": 4, "d_ff": 256, "ctx_len": 128
  })";
  CHECK(parse_spec_json(ok, s).empty());
  CHECK(s.name == "teste");
  CHECK(s.cfg.n_layers == 2 && s.cfg.d_model == 64 && s.cfg.n_heads == 4);
  CHECK(s.cfg.d_ff == 256 && s.cfg.ctx_len == 128);

  ModelSpec t;
  CHECK_ERR(parse_spec_json(R"({"n_layers": 2, "extra": 1})", t), "desconhecida");
  CHECK_ERR(parse_spec_json(R"({"n_layers": 2})", t), "obrigatoria");
  CHECK_ERR(parse_spec_json(R"({"n_layers": "dois"})", t), "inteiro");
  CHECK_ERR(parse_spec_json(R"({"n_layers": 2, "n_layers": 3})", t), "repetida");
  CHECK_ERR(parse_spec_json(R"({"n_layers": 99999999999})", t), "inteiro");
  CHECK(!parse_spec_json("", t).empty());
  CHECK(!parse_spec_json("[1, 2]", t).empty());
  CHECK(!parse_spec_json(R"({"n_layers": 2,})", t).empty());
  CHECK(!parse_spec_json(ok + " lixo", t).empty());

  const std::string bpe = R"({"tokenizer": "bpe", "n_layers": 1, "d_model": 8,
    "n_heads": 1, "d_ff": 8, "ctx_len": 8})";
  CHECK_ERR(parse_spec_json(bpe, t), "nao suportado");

  CHECK(!load_spec_json("/caminho/que/nao/existe.json", t).empty());
}

int main() {
  test_crc32();
  test_vocab_and_params();
  test_validate_config();
  test_roundtrip_memory();
  test_roundtrip_file();
  test_empty_model();
  test_bad_files();
  test_serialize_rejects_bad_models();
  test_spec_json();

  if (g_fail == 0)
    std::printf("OK: %d verificacoes passaram\n", g_checks);
  else
    std::printf("FALHARAM %d de %d verificacoes\n", g_fail, g_checks);
  return g_fail == 0 ? 0 : 1;
}

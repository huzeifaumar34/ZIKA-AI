#include "zika/spec.hpp"

#include <cctype>
#include <fstream>
#include <iterator>
#include <set>

namespace zika {

namespace {

struct Parser {
  const std::string &s;
  size_t i = 0;

  explicit Parser(const std::string &text) : s(text) {}

  void skip_ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      i++;
  }
  bool eat(char c) {
    skip_ws();
    if (i < s.size() && s[i] == c) {
      i++;
      return true;
    }
    return false;
  }
  // Texto entre aspas, sem sequências de escape.
  bool string(std::string &out) {
    skip_ws();
    if (i >= s.size() || s[i] != '"')
      return false;
    i++;
    out.clear();
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' || s[i] == '\n')
        return false;
      out.push_back(s[i++]);
    }
    if (i >= s.size())
      return false;
    i++;
    return true;
  }
  // Inteiro não negativo que cabe em 32 bits.
  bool number(uint32_t &out) {
    skip_ws();
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
      return false;
    uint64_t v = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
      v = v * 10 + static_cast<uint64_t>(s[i] - '0');
      if (v > 0xFFFFFFFFull)
        return false;
      i++;
    }
    out = static_cast<uint32_t>(v);
    return true;
  }
  std::string where() const { return " (perto do caracter " + std::to_string(i) + ")"; }
};

} // namespace

std::string parse_spec_json(const std::string &text, ModelSpec &spec) {
  Parser p(text);
  ModelSpec out;
  std::set<std::string> seen;

  if (!p.eat('{'))
    return "JSON invalido: esperava '{'" + p.where();

  bool first = true;
  while (true) {
    if (p.eat('}'))
      break;
    if (!first && !p.eat(','))
      return "JSON invalido: esperava ',' ou '}'" + p.where();
    first = false;

    std::string key;
    if (!p.string(key))
      return "JSON invalido: esperava o nome de uma chave" + p.where();
    if (!p.eat(':'))
      return "JSON invalido: esperava ':' depois de \"" + key + "\"" + p.where();
    if (!seen.insert(key).second)
      return "chave repetida: " + key;

    if (key == "name" || key == "tokenizer") {
      std::string v;
      if (!p.string(v))
        return "o valor de \"" + key + "\" tem de ser texto" + p.where();
      (key == "name" ? out.name : out.tokenizer) = v;
    } else {
      uint32_t *field = nullptr;
      if (key == "n_layers")
        field = &out.cfg.n_layers;
      else if (key == "d_model")
        field = &out.cfg.d_model;
      else if (key == "n_heads")
        field = &out.cfg.n_heads;
      else if (key == "d_ff")
        field = &out.cfg.d_ff;
      else if (key == "ctx_len")
        field = &out.cfg.ctx_len;
      else
        return "chave desconhecida: " + key;
      if (!p.number(*field))
        return "o valor de \"" + key + "\" tem de ser um inteiro nao negativo" +
               p.where();
    }
  }

  p.skip_ws();
  if (p.i != text.size())
    return "JSON invalido: texto a mais depois do '}'" + p.where();

  for (const char *req : {"n_layers", "d_model", "n_heads", "d_ff", "ctx_len"}) {
    if (!seen.count(req))
      return std::string("falta a chave obrigatoria: ") + req;
  }
  if (out.tokenizer != "char" && out.tokenizer != "bpe")
    return "tokenizer '" + out.tokenizer + "' desconhecido (opcoes: \"char\", \"bpe\")";

  spec = std::move(out);
  return "";
}

std::string load_spec_json(const std::string &path, ModelSpec &spec) {
  std::ifstream f(path);
  if (!f)
    return "nao consegui abrir " + path;
  const std::string text((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  return parse_spec_json(text, spec);
}

} // namespace zika

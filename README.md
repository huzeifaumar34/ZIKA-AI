# ZIKA-AI

Uma mini IA de conversa em português, construída **do zero**: motor de treino e inferência em C++, ferramentas em Python e um chat web que corre o modelo no próprio dispositivo.

> **Estado: planeamento.** Ainda não há código. Este repositório contém, por agora, o plano e a documentação. As fases estão em [`docs/PLANO.md`](docs/PLANO.md).

## O que é

O ZIKA-AI é um pequeno modelo de linguagem do tipo GPT (transformer *decoder-only*) que gera texto e conversa em português (de Portugal ou do Brasil), sobre conversa geral.

Não vai ser o ChatGPT. O objetivo é aprender e controlar tudo: dos tensores à interface de chat, sem PyTorch nem outras bibliotecas de IA. Com poucos dados e poucos parâmetros, o modelo escreve português plausível e responde ao básico, mas não "sabe" coisas como os grandes modelos.

## Arquitetura

| Parte | Linguagem | Função |
|---|---|---|
| `core/` | C++ | Tensores, matmul (AVX2 + OpenMP), transformer com *backprop* manual, AdamW, treino e inferência |
| `cli/` | C++ | Executável `zika` (`train`, `chat`, `export`) |
| `python/` | Python | Preparação de dados, tokenizer, avaliação, ligação ao núcleo por `ctypes` |
| `web/` | HTML + JavaScript | Chat que corre o modelo no navegador, offline, lendo o ficheiro `.zika` |

## Estrutura prevista

```
ZIKA-AI/
  core/     (C++: ops, transformer, treino, inferência)
  cli/      (executável zika)
  python/   (dados, tokenizer, bindings)
  web/      (chat HTML + JS)
  data/     (diálogos e texto de treino)
  models/   (pesos .zika)
  tests/    (incluindo testes de gradiente)
  docs/     (plano e notas)
  CMakeLists.txt
```

## Modelos

Tudo é definido por ficheiros de configuração, por isso mudar de tamanho não obriga a mudar código.

| Modelo | Camadas | Dimensão | Cabeças | Contexto | Parâmetros | Objetivo |
|---|---|---|---|---|---|---|
| ZIKA-Nano | 3 | 128 | 4 | 256 | ~0,6 M | Validar o motor, treina em minutos |
| ZIKA-Mini | 4 | 256 | 8 | 256 | ~3,5 M | O modelo "real", treina em horas |

Tokenizer: começa letra a letra e passa a BPE (~1000 tokens) mais tarde.

## Formato do chat

O modelo é treinado com diálogos neste formato:

```
<|user|> Olá, tudo bem? <|ia|> Olá! Tudo bem, e contigo? <|fim|>
```

## Fases

- [ ] 1. Esqueleto do projeto, CMake e formato `.zika`
- [ ] 2. Núcleo C++ com testes de gradiente
- [ ] 3. Dados e tokenizer
- [ ] 4. Treinar o ZIKA-Nano
- [ ] 5. Chat no terminal
- [ ] 6. Chat web
- [ ] 7. Treinar o ZIKA-Mini e quantizar (int8)

## Como vai ser usado (previsto)

Estes comandos ainda **não existem**; mostram a direção:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/zika train --config configs/nano.json --data data/dialogos.jsonl
./build/zika chat --model models/zika-nano.zika
```

## Licença

[MIT](LICENSE)

# ZIKA-AI

Uma mini IA de conversa em português, construída **do zero**: motor de treino e inferência em C++, ferramentas em Python e um chat web que corre o modelo no próprio dispositivo.

> **Estado: fase 1 concluída de 7.** Já existem o esqueleto em C++, o formato de ficheiro `.zika` e o CLI mínimo (`version`, `init`, `info`). O treino e o chat ainda **não existem**. O plano completo está em [`docs/PLANO.md`](docs/PLANO.md).

## O que é

O ZIKA-AI é um pequeno modelo de linguagem do tipo GPT (transformer *decoder-only*) que gera texto e conversa em português (de Portugal ou do Brasil), sobre conversa geral.

Não vai ser o ChatGPT. O objetivo é aprender e controlar tudo: dos tensores à interface de chat, sem PyTorch nem outras bibliotecas de IA. Com poucos dados e poucos parâmetros, o modelo escreve português plausível e responde ao básico, mas não "sabe" coisas como os grandes modelos.

## Arquitetura

| Parte | Linguagem | Função |
|---|---|---|
| `core/` | C++ | Tensores, matmul (AVX2 + OpenMP), transformer com *backprop* manual, AdamW, treino e inferência |
| `cli/` | C++ | Executável `zika` (hoje: `version`, `init`, `info`; previsto: `train`, `chat`, `export`) |
| `python/` | Python | Preparação de dados, tokenizer, avaliação, ligação ao núcleo por `ctypes` |
| `web/` | HTML + JavaScript | Chat que corre o modelo no navegador, offline, lendo o ficheiro `.zika` |

## Estrutura prevista

```
ZIKA-AI/
  core/     (C++: ops, transformer, treino, inferência)
  cli/      (executável zika)
  python/   (dados, tokenizer, bindings)
  web/      (chat HTML + JS)
  configs/  (nano.json, mini.json)
  data/     (diálogos e texto de treino)
  models/   (pesos .zika, não vão para o Git)
  tests/    (formato hoje; testes de gradiente na fase 2)
  docs/     (plano e notas)
  CMakeLists.txt
```

`python/`, `web/` e `data/` aparecem nas fases 3 a 6.

## Modelos

Tudo é definido por ficheiros de configuração, por isso mudar de tamanho não obriga a mudar código.

| Modelo | Camadas | Dimensão | Cabeças | Contexto | Parâmetros | Objetivo |
|---|---|---|---|---|---|---|
| ZIKA-Nano | 3 | 128 | 4 | 256 | 0,64 M | Validar o motor, treina em minutos |
| ZIKA-Mini | 4 | 256 | 8 | 256 | 3,26 M | O modelo "real", treina em horas |

Tokenizer: começa letra a letra (124 tokens) e passa a BPE (~1000 tokens) mais tarde; com BPE o Mini sobe para ~3,5 M parâmetros.

## Formato do chat

O modelo é treinado com diálogos neste formato:

```
<|user|> Olá, tudo bem? <|ia|> Olá! Tudo bem, e contigo? <|fim|>
```

## Fases

- [x] 1. Esqueleto do projeto, CMake e formato `.zika`
- [ ] 2. Núcleo C++ com testes de gradiente
- [ ] 3. Dados e tokenizer
- [ ] 4. Treinar o ZIKA-Nano
- [ ] 5. Chat no terminal
- [ ] 6. Chat web
- [ ] 7. Treinar o ZIKA-Mini e quantizar (int8)

## Como usar

Precisa de um compilador C++17 e de CMake 3.16 ou superior. O OpenMP é usado se existir.

```bash
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure      # corre os testes

./build/zika init --config configs/nano.json --out models/zika-nano.zika
./build/zika info models/zika-nano.zika --tensors
```

`init` cria um modelo com os pesos a zero (o treino chega na fase 4). Por omissão compila-se com `-march=native`; para um binário portátil use `-DZIKA_NATIVE=OFF`.

Previsto para as próximas fases (**ainda não existe**):

```bash
./build/zika train --config configs/nano.json --data data/dialogos.jsonl
./build/zika chat --model models/zika-nano.zika
```

## Licença

[MIT](LICENSE)

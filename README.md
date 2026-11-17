# ZIKA-AI

Uma mini IA de conversa em português, construída **do zero**: motor de treino e inferência em C++, ferramentas de dados em Python e um chat web que correrá no próprio dispositivo.

> **Estado: fase 3 concluída de 7.** Já existem o formato .zika, as operações numéricas, o Transformer com backpropagation, AdamW, tokenizer de caracteres e pipeline de dados JSONL. O loop de treino ainda não existe; será implementado na Fase 4. O plano completo está em docs/PLANO.md.

## O que é

O ZIKA-AI é um pequeno modelo de linguagem do tipo GPT (Transformer decoder-only) orientado para conversas em português. O objetivo é aprender e controlar tudo: dos tensores à interface de chat, sem PyTorch, TensorFlow ou outras bibliotecas de IA.

Não será equivalente aos grandes assistentes comerciais. Com poucos parâmetros e dados, o modelo terá capacidade limitada, poderá repetir respostas e cometer erros.

## Arquitetura

| Parte | Linguagem | Função |
|---|---|---|
| core/ | C++ | Operações numéricas, Transformer, backpropagation e AdamW |
| cli/ | C++ | Executável zika; atualmente version, init e info |
| python/zika/ | Python padrão | Tokenizer, validação JSONL e preparação de dados |
| python/tests/ | Python padrão | Testes do tokenizer e do pipeline de dados |
| data/ | JSONL | Conversas originais para o primeiro corpus |
| web/ | HTML + JavaScript | Chat offline previsto para a Fase 6 |

## Estado das fases

- [x] 1. Esqueleto do projeto, CMake e formato .zika
- [x] 2. Núcleo C++ com testes de gradiente
- [x] 3. Dados e tokenizer de caracteres
- [ ] 4. Treino do ZIKA-Nano
- [ ] 5. Chat no terminal
- [ ] 6. Chat web
- [ ] 7. Treinar o ZIKA-Mini e quantizar (int8)

## Fase 3: dados e tokenizer

O tokenizer char-v0.1 contém exatamente 124 tokens, na mesma ordem do núcleo C++: três tokens especiais, quebra de linha, os 95 caracteres ASCII imprimíveis e 25 caracteres acentuados do português. A codificação e a descodificação são reversíveis; um caractere não suportado gera erro explícito em vez de ser apagado.

O corpus inicial em data/dialogos.jsonl contém 81 conversas originais. Cada linha tem um objeto JSON com turnos alternados entre user e ia, começando em user e terminando em ia. O corpus é pequeno e serve para validar o pipeline e iniciar os primeiros testes de treino, não para representar todo o conhecimento da língua.

Para preparar os dados, a partir da raiz do repositório:

    PYTHONPATH=python python -m zika.data --input data/dialogos.jsonl --output-dir data/processed

O comando cria:
- train.bin e val.bin: IDs de token uint32 little-endian;
- tokenizer.json: vocabulário e versão do tokenizer;
- manifest.json: contagens, divisão, seed e hashes SHA-256.

A divisão treino/validação é determinística e feita ao nível da conversa, evitando que turnos da mesma conversa apareçam nos dois conjuntos. Os artefactos gerados ficam fora do Git.

Para executar os testes Python:

    PYTHONPATH=python python -m unittest discover -s python/tests -v

O GitHub Actions executa os testes C++ e Python e prepara o corpus em cada push para main, pull request ou execução manual.

## Modelos

| Modelo | Camadas | Dimensão | Cabeças | Contexto | Parâmetros | Objetivo |
|---|---:|---:|---:|---:|---:|---|
| ZIKA-Nano | 3 | 128 | 4 | 256 | 643 712 | Validar o treino em CPU |
| ZIKA-Mini | 4 | 256 | 8 | 256 | 3 256 832 | Modelo maior, fase 7 |

## Compilar o núcleo C++

Precisa de um compilador C++17 e CMake 3.16 ou superior. OpenMP é usado quando existe.

    cmake -S . -B build
    cmake --build build --parallel
    ctest --test-dir build --output-on-failure

Criar e inspecionar um modelo vazio:

    ./build/zika init --config configs/nano.json --out models/zika-nano.zika
    ./build/zika info models/zika-nano.zika --tensors

O comando init cria pesos iniciais a zero, não treina o modelo. O treino real começa na Fase 4.

## Licença

MIT.

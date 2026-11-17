# ZIKA-AI

Uma mini IA de conversa em português, construída **do zero**: motor de treino e inferência em C++, ferramentas de dados em Python e um chat web que correrá no próprio dispositivo.

> **Estado:** fases 1, 2 e 3 concluídas. O motor de treino da Fase 4 está implementado e validado com testes de sobreajuste e retoma; a campanha de treino do ZIKA-Nano ainda tem de ser executada e avaliada. O plano está em docs/PLANO.md.

## O que é

O ZIKA-AI é um pequeno modelo de linguagem do tipo GPT (Transformer decoder-only) orientado para conversas em português. O objetivo é aprender e controlar tudo: dos tensores à interface de chat, sem PyTorch, TensorFlow ou outras bibliotecas de IA.

Não será equivalente aos grandes assistentes comerciais. Com poucos parâmetros e dados, o modelo terá capacidade limitada, poderá repetir respostas e cometer erros.

## Arquitetura

| Parte | Linguagem | Função |
|---|---|---|
| core/ | C++ | Operações numéricas, Transformer, backpropagation, AdamW e treino |
| cli/ | C++ | Executável zika: version, init, info e train |
| python/zika/ | Python padrão | Tokenizer, validação JSONL e preparação de dados |
| python/tests/ | Python padrão | Testes do tokenizer e do pipeline de dados |
| data/ | JSONL | Conversas originais para o primeiro corpus |
| web/ | HTML + JavaScript | Chat offline previsto para a Fase 6 |

## Estado das fases

- [x] 1. Esqueleto do projeto, CMake e formato .zika
- [x] 2. Núcleo C++ com testes de gradiente
- [x] 3. Dados e tokenizer de caracteres
- [ ] 4. Treino do ZIKA-Nano (motor implementado; campanha real em curso)
- [ ] 5. Chat no terminal
- [ ] 6. Chat web
- [ ] 7. Treinar o ZIKA-Mini e quantizar (int8)

## Dados e tokenizer

O tokenizer char-v0.1 contém exatamente 124 tokens, na mesma ordem do núcleo C++: três tokens especiais, quebra de linha, os 95 caracteres ASCII imprimíveis e 25 caracteres acentuados do português. A codificação e a descodificação são reversíveis; um caractere não suportado gera erro explícito em vez de ser apagado.

O corpus inicial em data/dialogos.jsonl contém 81 conversas originais. Cada linha tem um objeto JSON com turnos alternados entre user e ia, começando em user e terminando em ia. O corpus é pequeno e serve para validar o pipeline e iniciar os primeiros testes de treino, não para representar todo o conhecimento da língua.

Preparar os dados localmente:

    PYTHONPATH=python python -m zika.data --input data/dialogos.jsonl --output-dir data/processed

São criados train.bin e val.bin (IDs uint32 little-endian), tokenizer.json e manifest.json. Os dados processados não são versionados.

## Fase 4: motor de treino

O comando zika train executa treino causal em batches com janelas aleatórias do corpus, acumulação de gradientes, AdamW, warm-up linear e decaimento cosseno. Mede a cross-entropy de treino e validação sem calcular gradientes durante a avaliação.

Exemplo de treino do ZIKA-Nano:

    PYTHONPATH=python python -m zika.data --input data/dialogos.jsonl --output-dir data/processed
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel 2
    ./build/zika train \
      --config configs/nano.json \
      --data data/processed/train.bin \
      --val-data data/processed/val.bin \
      --out models/zika-nano.zika \
      --steps 500 --batch-size 1 --seq-len 128

O programa cria também models/zika-nano.zika.ckpt. O checkpoint contém os pesos, os momentos do AdamW, o número do passo, o estado aleatório e as assinaturas CRC32 dos dois conjuntos de dados. Assim, só pode ser retomado com o mesmo corpus:

    ./build/zika train \
      --config configs/nano.json \
      --data data/processed/train.bin \
      --val-data data/processed/val.bin \
      --out models/zika-nano.zika \
      --steps 1000 \
      --resume models/zika-nano.zika.ckpt

O argumento --steps é o passo total pretendido; ao retomar, tem de ser maior que o passo guardado. O modelo final é guardado em formato .zika; os checkpoints ficam fora do Git.

### Treino na nuvem, sem PC

No GitHub, abre **Actions → ZIKA-AI Train Nano → Run workflow**. Define o número total de passos, batch size, comprimento da sequência e seed. O artefacto zika-nano-training inclui o modelo, o checkpoint e o log. Para continuar um treino, usa o ID da execução anterior no campo resume_run_id; o checkpoint será descarregado automaticamente.

A execução usa CPU. Não pressupõe GPU, não usa modelos pré-treinados e não transforma automaticamente o pequeno corpus de exemplo num assistente de grande escala. Começa com poucos passos para medir o tempo e a evolução da loss antes de aumentar a campanha.

## Testes

Compilar e executar todos os testes C++:

    cmake -S . -B build
    cmake --build build --parallel
    ctest --test-dir build --output-on-failure

Testes Python:

    PYTHONPATH=python python -m unittest discover -s python/tests -v

O CI compila o núcleo, verifica os gradientes, testa que o modelo minúsculo consegue reduzir a loss e retomar de checkpoint, valida os dados e executa um treino curto do ZIKA-Nano para verificar a integração.

## Modelos

| Modelo | Camadas | Dimensão | Cabeças | Contexto | Parâmetros |
|---|---:|---:|---:|---:|---:|
| ZIKA-Nano | 3 | 128 | 4 | 256 | 643 712 |
| ZIKA-Mini | 4 | 256 | 8 | 256 | 3 256 832 |

## Licença

MIT.

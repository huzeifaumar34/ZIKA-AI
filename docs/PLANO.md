# Plano do ZIKA-AI

Documento de definição e implementação do projeto.

## 1. Decisões

- **Nome:** ZIKA-AI
- **Língua:** português de Portugal ou do Brasil, sem depender de regionalismos
- **Tema:** conversa geral
- **Licença:** MIT
- **Sem bibliotecas de IA:** nada de PyTorch, TensorFlow ou semelhantes. O núcleo usa a biblioteca padrão de C++ e OpenMP
- **Linguagens:** C++ (núcleo e CLI), Python (dados e ferramentas), JavaScript (chat web)

## 2. Arquitetura do modelo

Transformer decoder-only, no estilo GPT:

- Embeddings de tokens com pesos partilhados com a camada de saída (weight tying)
- Embeddings de posição aprendidos; RoPE fica como melhoria futura
- Blocos com LayerNorm antes da atenção e do MLP (pre-LN), atenção causal multi-cabeça e MLP com GELU
- Treino com AdamW, warm-up e decaimento cosseno, corte de gradiente
- Inferência com KV cache e amostragem por temperatura, top-k e top-p

### Configurações

| | ZIKA-Nano | ZIKA-Mini |
|---|---:|---:|
| Camadas | 3 | 4 |
| Dimensão (d_model) | 128 | 256 |
| Cabeças | 4 | 8 |
| d_ff | 512 | 1024 |
| Contexto | 256 | 256 |
| Vocabulário | 124 (caracteres) | 124 inicialmente; BPE futuro |
| Parâmetros | 643 712 | 3 256 832 |

Os tempos de treino reais serão medidos na Fase 4 e registados aqui.

## 3. Tokenizer

- **v0.1 (implementado):** caractere a caractere, com 124 IDs fixos: 3 tokens especiais, quebra de linha, 95 caracteres ASCII imprimíveis e 25 caracteres acentuados
- **v0.2 (futuro):** BPE com cerca de 1000 tokens, treinado em Python e guardado dentro do ficheiro .zika

Tokens especiais e IDs: 0 = <|user|>, 1 = <|ia|>, 2 = <|fim|>. Os restantes IDs seguem a ordem de default_char_vocab() em core/src/format.cpp. O tokenizer Python reproduz exatamente essa ordem. Não há normalização Unicode silenciosa; caracteres fora do vocabulário são rejeitados para preservar os dados.

## 4. Dados

- **Formato:** JSONL, uma conversa por linha:

      {"turns":[{"role":"user","text":"Olá!"},{"role":"ia","text":"Olá! Como posso ajudar?"}]}

- Os turnos alternam entre user e ia; cada conversa começa com user e termina com ia
- Corpus inicial original em data/dialogos.jsonl: 81 conversas de conversa geral, programação, tecnologia, aprendizagem e explicações simples
- Só se usam textos originais ou que temos direito de usar
- É possível acrescentar dados próprios em data/, desde que respeitem o esquema e o vocabulário do tokenizer
- Pré-treino em texto corrido antes do ajuste nos diálogos fica como opção futura

### Pipeline de dados (Fase 3 implementada)

    PYTHONPATH=python python -m zika.data --input data/dialogos.jsonl --output-dir data/processed

O pipeline valida JSONL, alternância de papéis, marcadores reservados e cobertura do vocabulário; serializa cada conversa com tokens especiais; divide treino e validação ao nível da conversa; exporta IDs uint32 little-endian e grava tokenizer.json e manifest.json com contagens e hashes SHA-256. Os artefactos processados não são versionados.

## 5. Formato de ficheiro .zika (versão 1, fixada na Fase 1)

Tudo em little-endian:

| Campo | Tipo | Notas |
|---|---|---|
| Magia | 4 bytes | ZIKA |
| Versão | u32 | 1 |
| Tipo de dados | u32 | 0 = F32 (único implementado), 1 = F16 e 2 = I8 reservados para a Fase 7 |
| Configuração | 6 x u32 | vocab_size, n_layers, d_model, n_heads, d_ff, ctx_len |
| Tipo de tokenizer | u32 | 0 = caracteres, 1 = BPE |
| Tokens | u32 n, depois n x (u32 tamanho + bytes) | n tem de ser igual a vocab_size |
| Regras BPE | u32 m, depois m x (u32, u32) | m = 0 para caracteres |
| Tensores | u32 n, depois n x (nome, dimensões, dados F32) | pela ordem abaixo |
| Checksum | u32 | CRC32 de todos os bytes anteriores |

Tensores, por ordem (a camada de saída reutiliza tok_emb):

    tok_emb [vocab, d]        pos_emb [ctx, d]
    por camada l = 0..n_layers-1:
      layer<l>.ln1.g [d]      layer<l>.ln1.b [d]
      layer<l>.attn.wqkv [d, 3d]   layer<l>.attn.bqkv [3d]
      layer<l>.attn.wo [d, d]      layer<l>.attn.bo [d]
      layer<l>.ln2.g [d]      layer<l>.ln2.b [d]
      layer<l>.mlp.w1 [d, d_ff]    layer<l>.mlp.b1 [d_ff]
      layer<l>.mlp.w2 [d_ff, d]    layer<l>.mlp.b2 [d]
    lnf.g [d]                 lnf.b [d]

O leitor rejeita ficheiros com magia ou versão erradas, checksum inválido, truncados, com tensores fora de ordem ou com dimensões que não batem com a configuração. O mesmo ficheiro será lido pelo C++ e pelo JavaScript.

Os ficheiros de configuração (configs/*.json) são JSON plano com as chaves name, tokenizer, n_layers, d_model, n_heads, d_ff e ctx_len. O tamanho do vocabulário vem do tokenizer.

## 6. Fases

| Fase | Entrega | Critério de conclusão |
|---|---|---|
| 1 (feita) | Esqueleto, CMake, formato .zika | Compila e escreve/lê um .zika vazio |
| 2 (feita) | Núcleo C++ (ops, transformer, AdamW) | Testes de gradiente passam no CI |
| 3 (feita) | Corpus, tokenizer char-v0.1, pipeline JSONL | Corpus gerado e tokenizado de ida e volta sem perdas |
| 4 (em curso) | Treino do ZIKA-Nano | Treino real do Nano, loss de validação acompanhada e amostra de texto avaliada |
| 5 | Chat no terminal | zika chat mantém uma conversa |
| 6 | Chat web | Abre no telemóvel e responde offline |
| 7 | ZIKA-Mini e int8 | Modelo maior treinado e exportado em tamanho reduzido |

### Implementação validada na Fase 2

- core/src/ops.cpp: multiplicação matricial, LayerNorm, GELU tanh-aproximada e softmax estável, cada uma com backward manual
- core/src/transformer.cpp: embeddings de token/posição, blocos pre-LN, atenção multi-cabeça causal, projeções QKV, MLP GELU, ligações residuais, LayerNorm final e weight tying entre tok_emb e a saída; inclui cross-entropy causal e gradientes dos parâmetros
- core/src/optimizer.cpp: AdamW com momentos, correção de bias, weight decay desacoplado (exclui biases e ganhos de LayerNorm) e clipping pela norma global
- initialize_weights: inicialização determinística das matrizes e embeddings; zika init continua a gerar pesos a zero para manter o comportamento documentado da Fase 1
- CI: compilação C++17, testes CTest e smoke test do CLI

### Implementação da Fase 3

- python/zika/tokenizer.py: CharTokenizer de 124 tokens; encode/decode sem perdas, marcadores especiais atómicos, rejeição de caracteres desconhecidos e gravação/leitura do vocabulário JSON
- python/zika/data.py: validação do esquema JSONL, alternância user/ia, serialização de conversas, divisão determinística por conversa e exportação de IDs uint32 little-endian
- data/dialogos.jsonl: 81 conversas originais para validar o pipeline e servir de corpus inicial
- python/tests/: testes unitários de vocabulário, acentos, round-trip, tokens inválidos, JSONL, divisão determinística e binários
- CI: testes C++ e Python e preparação automática do corpus em cada push, pull request ou execução manual

### Implementação da Fase 4 (motor e infraestrutura de treino)

- core/src/training.cpp: janelas causais aleatórias, batches, acumulação média de gradientes, avaliação determinística e tratamento de dados uint32 little-endian
- AdamW: learning rate ajustável por passo e serialização dos momentos para checkpoints
- Scheduler: warm-up linear seguido de decaimento cosseno até uma fração mínima da taxa inicial
- Checkpoint ZCKP v1: modelo .zika embutido, estado AdamW, contador de passos, estado do gerador aleatório, CRC32 dos dados de treino/validação e checksum global; escrita atómica e rejeição de corpus/configuração diferentes
- CLI: comando zika train com opções de passos, batch, comprimento de sequência, seed, hiperparâmetros, checkpoint e retoma
- CI: teste de sobreajuste num padrão artificial, teste de retoma e treino curto do modelo Nano para validar integração

A Fase 4 só será marcada como concluída depois de executar uma campanha de treino real do ZIKA-Nano, confirmar a evolução da loss e avaliar texto gerado. O pequeno conjunto de 81 conversas é apenas um corpus inicial, não uma base suficiente para um assistente abrangente.

## 7. Testes

- **Gradiente numérico:** matmul, LayerNorm, GELU e softmax são comparados com diferenças finitas; o teste integrado também verifica gradientes de embeddings, atenção, MLP e LayerNorm do Transformer
- **Sobreajuste:** o modelo minúsculo tem de reduzir a loss num padrão repetido; o teste também comprova retoma de checkpoint
- **Ida e volta:** texto -> IDs -> texto preserva os caracteres; as conversas serializadas preservam os marcadores especiais; a preparação é determinística
- **Dados:** registos inválidos, papéis fora de ordem, caracteres não suportados e ficheiros JSONL malformados são rejeitados
- **Formato .zika:** exportar e voltar a carregar dá exatamente os mesmos bytes; ficheiros corrompidos, truncados e com versão errada são rejeitados
- **Paridade:** C++ e JavaScript produzem os mesmos logits para a mesma entrada (critério da Fase 6)

## 8. Riscos e limites

- Com poucos parâmetros e poucos dados, o modelo erra factos e perde o fio à conversa
- O treino em CPU é lento; usam-se checkpoints para retomar e o mesmo código pode correr noutro computador
- O tokenizer char consome mais posições de contexto do que BPE; BPE fica para uma melhoria futura
- O corpus atual é pequeno e pedagógico; não deve ser interpretado como modelo já treinado nem como cobertura completa do português
- Os modelos pequenos podem repetir-se; a amostragem ajuda, mas não resolve essa limitação

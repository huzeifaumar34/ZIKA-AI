# Plano do ZIKA-AI

Documento de definicao e implementacao. O projeto privilegia reproducibilidade, proveniencia dos dados e avaliacao honesta sobre metricas isoladas.

## 1. Decisoes

- **Nome:** ZIKA-AI
- **Lingua:** portugues de Portugal e do Brasil, sem depender de regionalismos
- **Tema:** conversa geral
- **Licenca do codigo:** MIT
- **Dependencias de IA:** sem PyTorch, TensorFlow ou pesos pre-treinados; nucleo em C++ e ferramentas em Python
- **Linguagens:** C++ (nucleo e CLI), Python (dados e ferramentas), JavaScript (chat web)

## 2. Arquitetura

Transformer decoder-only:
- Embeddings de tokens com pesos partilhados com a camada de saida
- Embeddings posicionais aprendidos
- Blocos pre-LN, atencao causal multi-cabeca e MLP com GELU
- AdamW, warm-up, decaimento cosseno e clipping de gradiente
- Inferencia autoregressiva com temperatura, top-k e top-p
- KV cache como otimizacao futura

| | ZIKA-Nano | ZIKA-Mini |
|---|---:|---:|
| Camadas | 3 | 4 |
| d_model | 128 | 256 |
| Cabecas | 4 | 8 |
| d_ff | 512 | 1024 |
| Contexto | 256 | 256 |
| Vocabulario inicial | 124 | 124 |
| Parametros | 643.712 | 3.256.832 |

## 3. Tokenizer

- **char-v0.1:** 124 tokens fixos; codificacao reversivel e rejeicao explicita de caracteres desconhecidos.
- **BPE byte-v0.1 (implementacao inicial):** 3 tokens estruturais, 256 tokens de byte UTF-8 e merges deterministas aprendidos apenas no treino; byte fallback garante cobertura de Unicode. O tokenizer fica embutido no `.zika` e e exportado em `.zkt` para preparar dados/treino.
- O tokenizer atual e funcional mas ineficiente em contexto; BPE deve preceder a expansao agressiva do contexto.

## 4. Dados

Formato JSONL, uma conversa por linha, com turnos alternados `user` e `ia`. Cada conversa comeca no utilizador e termina na IA.

Corpus original atualmente dividido entre:
- `data/dialogos.jsonl`: corpus inicial.
- `data/dialogos_extra.jsonl`: dialogos originais adicionais, escritos para cobrir aprendizagem, ciencia, computacao, programacao, redes, seguranca e conceitos de IA.

O pipeline:
- aceita varios ficheiros JSONL;
- valida estrutura, alternancia, texto nao vazio, tokens reservados e cobertura do tokenizer;
- remove duplicados apos normalizacao Unicode NFKC, casefold e normalizacao de espacos;
- divide por conversa em treino, validacao e teste, com seed deterministica;
- produz binarios uint32 little-endian e manifesto com hashes SHA-256, contagens e fontes.

**Regra de dados:** nunca usar o conjunto de teste durante o treino ou para escolher hiperparametros. Registar a origem e a licenca de qualquer conjunto externo. O corpus atual permanece pequeno e nao sustenta a ambicao de um assistente geral.

## 5. Formato .zika

Formato de modelo v1, little-endian, com magic, configuracao, tokenizer, tensores F32 e CRC32. F16/I8 ficam reservados para fases futuras. A leitura valida versao, checksum, dimensoes e ordem dos tensores.

## 6. Fases

| Fase | Entrega | Criterio |
|---|---|---|
| 1 | Esqueleto, CMake e formato .zika | Leitura/escrita validada |
| 2 | Operacoes, Transformer e AdamW | Testes numericos de gradiente |
| 3 | Tokenizer e pipeline de dados | Ida-e-volta sem perdas e manifesto |
| 4A | Motor de treino inicial | Nano treinado e loss monitorizada |
| 4B | Robustez de treino | Scheduler retomavel, avaliacao eficiente, checkpoints verificaveis, testes com texto real |
| 4C | Dados e avaliacao | Tres conjuntos disjuntos, auditoria e avaliacao held-out |
| 5 | Chat no terminal | Inferencia autoregressiva, amostragem, historico e avaliacao qualitativa |
| 6 | Chat web offline | Paridade C++/JavaScript e uso no telemovel |
| 7 | ZIKA-Mini e quantizacao | Benchmark de qualidade, memoria e velocidade |

## 7. Robustez do treino

### Scheduler e checkpoints
- ZCKP v2 guarda estado do scheduler, taxa atual, horizonte, warm-up, parametros relevantes, pesos, AdamW, RNG e hashes dos dados.
- Prolongar um treino preserva a taxa atual e cria uma transicao suave; nao pode existir salto devido a um novo `--steps`.
- Retoma rejeita alteracoes silenciosas de corpus ou parametros que alterem a atualizacao.
- ZCKP v1 so pode ser retomado quando o utilizador indica explicitamente o horizonte original via `--resume-legacy-total-steps`.

### Avaliacao
- Avaliacoes periodicas usam janelas deterministicas distribuidas no corpus e limitadas por `eval_tokens`.
- Avaliacao completa acontece em intervalos maiores e no fim.
- O conjunto de teste e avaliado depois do treino, por comando separado; nunca participa no scheduler.

### Integridade
- Deteccao de pesos NaN/Inf.
- Escrita atomica de checkpoints, modelos e artefactos de dados.
- Hashes de corpus e configuracao no checkpoint.
- Logs distinguem amostras de avaliacao e avaliacao completa.

## 8. Testes

- Gradientes numericos de operacoes e Transformer
- Serializacao e corrupcao de modelos/checkpoints
- Tokenizer e pipeline JSONL com caracteres suportados e rejeicao de caracteres invalidos
- Deduplicacao, determinismo, tres-way split e hashes
- Treino num corpus dialogico real (nao padrao alternado artificial)
- Reducao de loss de treino e verificacao de validacao
- Avaliacao amostrada vs completa
- Extensao de scheduler sem salto, incompatibilidade de corpus/configuracao
- Smoke test do CLI e avaliacao em dados held-out no CI

## 9. Limites e riscos conhecidos

- 643 mil parametros e corpus pequeno nao competem com assistentes comerciais.
- Loss menor nao prova factualidade, utilidade, seguranca nem coerencia.
- Tokenizer de caracteres desperdiça contexto; BPE e uma prioridade.
- Treino em CPU e viavel para o Nano, mas modelos maiores podem exigir muito tempo.
- Dados externos exigem verificacao de licenca, qualidade, duplicados e contaminacao do teste.
- Testes automaticos precisam ser complementados por uma bateria qualitativa e revisao humana.

## 10. Tokenizer BPE byte-level (implementacao inicial)\n\n- `python/zika/tokenizer.py`: BPE deterministico com 3 tokens especiais, 256 bytes base, byte fallback e merges ordenados.\n- `python/zika/data.py`: aprende merges apenas no conjunto de treino, depois da divisao; exporta JSON, tokenizer ZTKZ v1 e tres conjuntos.\n- `core/src/format.cpp`: le e valida magic, versao, CRC32, ordem dos bytes e integridade dos merges; `.zika` continua a embutir os tokens e merges.\n- `core/src/generation.cpp`: encode/decode UTF-8 byte-level e inferencia com BPE; texto literal que se pareca com marcador especial nao e interpretado como token estrutural.\n- `configs/nano-bpe.json` e `--tokenizer-file` permitem treinar/inferir BPE mantendo char como default.\n- CI testa BPE round-trip, checksum, CLI train/eval/chat. A qualidade do Nano BPE continua por medir em campanha real.\n\n## 11. Inferencia e chat no terminal (implementacao inicial)

- `core/src/generation.cpp`: codificacao e descodificacao UTF-8 para char-v0.1, amostragem por temperatura/Top-K/Top-P e geracao autoregressiva atraves de `forward_logits`.
- `cli/chat.cpp`: `zika chat --model ...`, historico limitado a `ctx_len`, encerramento por `/sair` e parametros de amostragem.
- `tests/test_generation.cpp`: ida-e-volta de texto, rejeicao de UTF-8/caracteres desconhecidos, Top-K, parametros e geracao.
- Limite conhecido: cada token recalcula toda a janela; KV cache fica como otimizacao posterior.
- Critério de qualidade pendente: testar com o Nano efetivamente treinado, recolher exemplos de respostas e avaliar coerencia, repeticao e utilidade. O CI verifica funcionamento, nao qualidade semantica.

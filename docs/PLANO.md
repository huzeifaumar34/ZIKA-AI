# Plano do ZIKA-AI

Documento de definição do projeto. Os pontos marcados como **rascunho** são fixados na fase indicada.

## 1. Decisões

- **Nome:** ZIKA-AI
- **Língua:** português de Portugal ou do Brasil (texto em português padrão, sem depender de regionalismos)
- **Tema:** conversa geral
- **Licença:** MIT
- **Sem bibliotecas de IA:** nada de PyTorch, TensorFlow ou semelhantes. O núcleo usa só a biblioteca padrão de C++ e OpenMP
- **Linguagens:** C++ (núcleo e CLI), Python (dados e ferramentas), JavaScript (chat web)

## 2. Arquitetura do modelo

Transformer *decoder-only*, no estilo GPT:

- Embeddings de tokens com pesos partilhados com a camada de saída (*weight tying*)
- Embeddings de posição aprendidos (rascunho; RoPE fica como melhoria futura)
- Blocos com LayerNorm antes da atenção e do MLP (*pre-LN*), atenção causal multi-cabeça e MLP com GELU
- Treino com AdamW, *warm-up* + decaimento cosseno, corte de gradiente
- Inferência com *KV cache* e amostragem por temperatura, top-k e top-p

### Configurações

| | ZIKA-Nano | ZIKA-Mini |
|---|---|---|
| Camadas | 3 | 4 |
| Dimensão (`d_model`) | 128 | 256 |
| Cabeças | 4 | 8 |
| `d_ff` | 512 | 1024 |
| Contexto | 256 | 256 |
| Vocabulário | 124 (letras) | 124 (letras), depois ~1000 (BPE) |
| Parâmetros | 643 712 | 3 256 832 (~3,5 M com BPE) |

Os tempos de treino reais serão medidos na fase 4 e registados aqui.

## 3. Tokenizer

1. **v0.1:** letra a letra (124 tokens: 3 especiais, quebra de linha, ASCII imprimível e 25 letras acentuadas), simples e sem dicionário
2. **v0.2:** BPE com cerca de 1000 tokens, treinado em Python e guardado dentro do ficheiro `.zika`

Tokens especiais: `<|user|>`, `<|ia|>`, `<|fim|>`.

## 4. Dados

- Formato: JSONL, uma conversa por linha
  ```json
  {"turns": [{"role": "user", "text": "Olá!"}, {"role": "ia", "text": "Olá! Como posso ajudar?"}]}
  ```
- Conversas de conversa geral: saudações, perguntas simples, opiniões leves, histórias curtas, pequenas explicações
- Dados gerados por script e escritos à mão; é possível juntar dados próprios em `data/`
- Só se usam textos que temos direito a usar (originais ou de domínio público)
- Opcional mais tarde: pré-treino em texto corrido em português antes do ajuste nos diálogos

## 5. Formato de ficheiro `.zika` (versão 1, fixada na fase 1)

Tudo em *little-endian*:

| Campo | Tipo | Notas |
|---|---|---|
| Magia | 4 bytes | `ZIKA` |
| Versão | u32 | 1 |
| Tipo de dados | u32 | 0 = F32 (único implementado), 1 = F16 e 2 = I8 reservados para a fase 7 |
| Configuração | 6 × u32 | `vocab_size`, `n_layers`, `d_model`, `n_heads`, `d_ff`, `ctx_len` |
| Tipo de tokenizer | u32 | 0 = letras, 1 = BPE |
| Tokens | u32 `n`, depois `n` × (u32 tamanho + bytes) | `n` tem de ser igual a `vocab_size` |
| Regras BPE | u32 `m`, depois `m` × (u32, u32) | `m` = 0 para letras |
| Tensores | u32 `n`, depois `n` × (u32 tamanho do nome, nome, u32 `ndim`, `ndim` × u32 dimensões, dados F32) | pela ordem abaixo |
| Checksum | u32 | CRC32 de todos os bytes anteriores |

Tensores, por ordem (a camada de saída reutiliza `tok_emb`):

```
tok_emb [vocab, d]        pos_emb [ctx, d]
por camada l = 0..n_layers-1:
  layer<l>.ln1.g [d]      layer<l>.ln1.b [d]
  layer<l>.attn.wqkv [d, 3d]   layer<l>.attn.bqkv [3d]
  layer<l>.attn.wo [d, d]      layer<l>.attn.bo [d]
  layer<l>.ln2.g [d]      layer<l>.ln2.b [d]
  layer<l>.mlp.w1 [d, d_ff]    layer<l>.mlp.b1 [d_ff]
  layer<l>.mlp.w2 [d_ff, d]    layer<l>.mlp.b2 [d]
lnf.g [d]                 lnf.b [d]
```

O leitor rejeita ficheiros com magia ou versão erradas, checksum inválido, truncados, com tensores fora de ordem ou com dimensões que não batem com a configuração. O mesmo ficheiro será lido pelo C++ e pelo JavaScript.

Os ficheiros de configuração (`configs/*.json`) são JSON plano com as chaves `name`, `tokenizer`, `n_layers`, `d_model`, `n_heads`, `d_ff` e `ctx_len`. O tamanho do vocabulário vem do tokenizer.

## 6. Fases

| Fase | Entrega | Critério de conclusão |
|---|---|---|
| 1 (feita) | Esqueleto, CMake, formato `.zika` | Compila e escreve/lê um `.zika` vazio |
| 2 | Núcleo C++ (ops, transformer, AdamW) | Testes de gradiente passam |
| 3 | Dados e tokenizer | Corpus gerado e tokenizado de ida e volta sem perdas |
| 4 | Treino do ZIKA-Nano | A perda desce e o texto gerado parece português |
| 5 | Chat no terminal | `zika chat` mantém uma conversa |
| 6 | Chat web | Abre no telemóvel e responde offline |
| 7 | ZIKA-Mini e int8 | Modelo maior treinado e exportado em tamanho reduzido |

## 7. Testes

- **Gradiente numérico:** cada operação com *backprop* é comparada com diferenças finitas
- **Sobreajuste:** o modelo tem de conseguir decorar um conjunto minúsculo antes de treinar a sério
- **Ida e volta:** exportar para `.zika` e voltar a carregar dá exatamente os mesmos bytes (já testado na fase 1, incluindo ficheiros corrompidos, truncados e com versão errada; os testes também correm com ASan e UBSan)
- **Paridade:** C++ e JavaScript produzem os mesmos *logits* para a mesma entrada

## 8. Riscos e limites

- Com poucos parâmetros e poucos dados, o modelo erra factos e perde o fio à conversa
- O treino em CPU é lento; usam-se *checkpoints* para retomar e o mesmo código pode correr noutro computador
- Os modelos pequenos repetem-se; a amostragem (temperatura, top-k, top-p) ajuda, mas não resolve

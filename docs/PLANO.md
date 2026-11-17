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
| Vocabulário | ~100 (letras) | ~1000 (BPE) |
| Parâmetros (aprox.) | 0,6 M | 3,5 M |

Os tempos de treino reais serão medidos na fase 4 e registados aqui.

## 3. Tokenizer

1. **v0.1:** letra a letra (vocabulário de cerca de 100 caracteres), simples e sem dicionário
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

## 5. Formato de ficheiro `.zika` (rascunho, fixado na fase 1)

1. Magia `ZIKA` e versão do formato
2. Configuração (vocabulário, camadas, dimensão, cabeças, `d_ff`, contexto)
3. Tipo de dados dos pesos: float32, float16 ou int8
4. Vocabulário e regras do tokenizer
5. Tensores por ordem fixa, em *little-endian*

O mesmo ficheiro é lido pelo C++ e pelo JavaScript.

## 6. Fases

| Fase | Entrega | Critério de conclusão |
|---|---|---|
| 1 | Esqueleto, CMake, formato `.zika` | Compila e escreve/lê um `.zika` vazio |
| 2 | Núcleo C++ (ops, transformer, AdamW) | Testes de gradiente passam |
| 3 | Dados e tokenizer | Corpus gerado e tokenizado de ida e volta sem perdas |
| 4 | Treino do ZIKA-Nano | A perda desce e o texto gerado parece português |
| 5 | Chat no terminal | `zika chat` mantém uma conversa |
| 6 | Chat web | Abre no telemóvel e responde offline |
| 7 | ZIKA-Mini e int8 | Modelo maior treinado e exportado em tamanho reduzido |

## 7. Testes

- **Gradiente numérico:** cada operação com *backprop* é comparada com diferenças finitas
- **Sobreajuste:** o modelo tem de conseguir decorar um conjunto minúsculo antes de treinar a sério
- **Ida e volta:** exportar para `.zika` e voltar a carregar dá as mesmas saídas
- **Paridade:** C++ e JavaScript produzem os mesmos *logits* para a mesma entrada

## 8. Riscos e limites

- Com poucos parâmetros e poucos dados, o modelo erra factos e perde o fio à conversa
- O treino em CPU é lento; usam-se *checkpoints* para retomar e o mesmo código pode correr noutro computador
- Os modelos pequenos repetem-se; a amostragem (temperatura, top-k, top-p) ajuda, mas não resolve

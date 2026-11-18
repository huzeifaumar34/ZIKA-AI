# ZIKA-AI

Um pequeno modelo de linguagem em portugues, construido de raiz: Transformer decoder-only, motor numerico e treino em C++, ferramentas de dados em Python e chat local. O projeto nao usa PyTorch, TensorFlow nem pesos pre-treinados.

> **Estado:** o nucleo Transformer e o pipeline de dados estao implementados. O ZIKA-Nano executou uma campanha inicial de 500 passos, reduzindo a loss de validacao de 4,815 para 2,527. Essa medicao nao demonstra, por si so, qualidade conversacional. A branch `engineering/phase4-hardening` reforca scheduler/checkpoints, avaliacao, dados e testes antes de nova campanha de treino.

## Arquitetura atual

| Componente | Funcao |
|---|---|
| `core/` | Operacoes tensoriais, Transformer, backpropagation, AdamW e treino |
| `cli/` | Comandos `zika init`, `info`, `train`, `eval` e, na fase 5, `chat` |
| `python/zika/` | Tokenizer, validacao, deduplicacao e preparacao do corpus |
| `data/` | Dialogos originais em JSONL |
| `.github/workflows/` | CI e treino reproduzivel em CPU no GitHub Actions |

### ZIKA-Nano

- 3 camadas Transformer, dimensao 128, 4 cabecas, d_ff 512.
- 643.712 parametros.
- Contexto 256.
- Tokenizer char-v0.1 com 124 tokens.
- Pesos F32; BPE, KV cache e quantizacao ficam para fases posteriores.

O tokenizer de caracteres e simples e reversivel, mas e ineficiente: 256 tokens representam aproximadamente 256 caracteres. A transicao para BPE e uma melhoria prioritaria antes de aumentar muito o contexto.

## Dados

O corpus e composto por textos originais escritos para o projeto. O pipeline aceita um ou mais ficheiros JSONL com uma conversa por linha:

```json
{"turns":[{"role":"user","text":"O que e uma API?"},{"role":"ia","text":"E uma interface para programas comunicarem."}]}
```

Preparar tres conjuntos disjuntos por conversa:

```bash
PYTHONPATH=python python -m zika.data \
  --input data/dialogos.jsonl data/dialogos_extra.jsonl \
  --output-dir data/processed \
  --validation-ratio 0.10 \
  --test-ratio 0.10
```

A ferramenta valida papéis, alternancia, marcadores reservados e cobertura do vocabulario; remove duplicados normalizados; separa treino, validacao e teste; exporta `train.bin`, `val.bin` e `test.bin`; e escreve um manifesto com hashes SHA-256, contagens e origem dos ficheiros. A divisao e deterministica para a mesma seed.

Os dados atuais continuam a ser um corpus pequeno de arranque. Nao devem ser descritos como suficientes para ensinar conhecimento geral. Para ampliar o corpus, acrescenta dialogos originais ou dados com licenca compativel e regista a proveniencia. Nao mistures o conjunto de teste no treino.

## Treino na nuvem

O computador local nao e necessario para executar a campanha Nano. No GitHub, abre **Actions → ZIKA-AI Train Nano → Run workflow**. O workflow compila C++, prepara os tres conjuntos, treina em CPU, avalia o modelo no conjunto de teste isolado e publica modelo, checkpoint, logs e manifesto como artefacto.

Exemplo equivalente no terminal:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DZIKA_NATIVE=OFF
cmake --build build --parallel 2

PYTHONPATH=python python -m zika.data \
  --input data/dialogos.jsonl data/dialogos_extra.jsonl \
  --output-dir data/processed

./build/zika train \
  --config configs/nano.json \
  --data data/processed/train.bin \
  --val-data data/processed/val.bin \
  --out models/zika-nano.zika \
  --steps 1000 --batch-size 1 --seq-len 128 \
  --eval-every 100 --eval-tokens 2048 --full-eval-every 500

./build/zika eval \
  --model models/zika-nano.zika \
  --data data/processed/test.bin \
  --seq-len 128
```

As avaliacoes periodicas usam janelas deterministicamente distribuidas ao longo dos conjuntos; a avaliacao final percorre todo o conjunto. A validacao pode acompanhar decisoes de treino. O teste reservado serve apenas para medir o modelo final.

### Retoma de treino

Os checkpoints ZCKP v2 guardam pesos, estado do AdamW, RNG, estado do scheduler e os parametros relevantes do treino. Prolongar uma campanha cria uma transicao suave a partir da taxa de aprendizagem atual; nao recalcula o scheduler como se o modelo estivesse no inicio.

Exemplo:

```bash
./build/zika train ... --steps 1500 --resume models/zika-nano.zika.ckpt
```

A retoma exige o mesmo corpus, batch size, seq-len, learning rate, weight decay e grad clip. Para checkpoints antigos ZCKP v1, que nao guardavam o horizonte do scheduler, e obrigatorio indicar o total original de passos, por exemplo `--resume-legacy-total-steps 500`. Sem esse valor, a retoma e rejeitada em vez de adivinhar um scheduler.

## Avaliacao e testes

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

Os testes incluem gradientes numericos, formato/checksum, tokenizacao sem perdas, validacao e deduplicacao de JSONL, tres conjuntos separados, treino com texto dialogico real, avaliacao amostrada, extensao suave de scheduler e rejeicao de checkpoint com corpus/configuracao diferentes. O CI tambem treina o Nano por poucos passos e avalia num conjunto de teste isolado.

Passar os testes verifica a integridade do codigo e do pipeline; nao substitui uma avaliacao humana de respostas nem comprova factualidade.

## Proximas prioridades

1. Reunir um corpus significativamente maior, diversificado, deduplicado e com proveniencia/licenca registada.
2. Treinar e avaliar o Nano com exemplos de teste nunca vistos.
3. Implementar tokenizer BPE e medir a melhoria de contexto.
4. Validar inferencia/chat no terminal e otimizar com KV cache.
5. Implementar chat web offline e so depois testar Mini e quantizacao.

## Licenca

O codigo do projeto usa a licenca MIT. A proveniencia e os termos dos dados devem ser verificados separadamente; uma licenca de codigo nao concede automaticamente direitos sobre dados externos.

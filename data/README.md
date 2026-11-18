# Dados do ZIKA-AI

Os ficheiros `dialogos*.jsonl` contem dialogos originais escritos para desenvolvimento e validacao inicial do pipeline. Nao sao um corpus de escala suficiente para treinar um assistente geral.

## Formato

Uma conversa completa por linha:
```json
{"turns":[{"role":"user","text":"Pergunta"},{"role":"ia","text":"Resposta"}]}
```
Os turnos alternam entre `user` e `ia`; a conversa termina sempre com `ia`.

## Adicionar dados

- Usa conteudo original ou dados cuja licenca permita explicitamente o uso pretendido.
- Regista a fonte, a licenca e eventuais atribuicoes fora do texto da conversa.
- Nao incluas dados pessoais, segredos, credenciais ou material privado.
- Evita respostas copiadas, duplicadas, contraditorias ou sem contexto.
- Mantem exemplos variados, factualmente verificaveis e linguisticamente naturais.
- Nao uses exemplos do conjunto de teste para ajustar o modelo.

## Preparacao

```bash
PYTHONPATH=python python -m zika.data \
  --input data/dialogos.jsonl data/dialogos_extra.jsonl \
  --output-dir data/processed \
  --validation-ratio 0.10 --test-ratio 0.10 --seed 42
```

O manifesto regista os hashes SHA-256 dos ficheiros fonte e dos tres conjuntos. A separacao e feita por conversa e a deduplicacao ocorre antes da divisao.

## Tokenizer BPE byte-level

O pipeline aceita `--tokenizer bpe --vocab-size 512`. O tokenizer comeca com os tres marcadores estruturais e 256 tokens correspondentes a bytes UTF-8; depois aprende merges deterministas a partir apenas do conjunto de treino. O fallback de bytes preserva Unicode fora do corpus.

O artefacto `tokenizer.zkt` usa magic `ZTKZ`, versao 1, tipo BPE, tokens, merges e CRC32. O ficheiro `tokenizer.json` e uma representacao legivel para auditoria. Ambos devem ser guardados junto do manifesto e do modelo para reproducibilidade. O modelo `.zika` embute os tokens e merges para que a inferencia nao dependa de ficheiros externos.

Char continua a ser o padrao e mantem a compatibilidade com modelos antigos. Nao mistures IDs de tokenizers diferentes, mesmo que provenham do mesmo texto.

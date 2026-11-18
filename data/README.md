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

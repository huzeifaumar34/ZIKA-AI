# Dados de exemplo do ZIKA-AI

O ficheiro dialogos.jsonl e um corpus inicial escrito para este projeto. Os exemplos sao originais e destinam-se a validar o pipeline da Fase 3; nao representam uma base de conhecimento abrangente.

## Formato

Cada linha contem uma conversa JSON com uma lista turns. Os papeis alternam entre user e ia, com inicio em user e fim em ia. O texto e preservado sem normalizacao e tem de ser representavel pelo tokenizer char-v0.1.

## Preparacao

Na raiz do repositorio, executa:

    PYTHONPATH=python python -m zika.data --input data/dialogos.jsonl --output-dir data/processed

O script cria train.bin, val.bin, tokenizer.json e manifest.json. Os binarios contem IDs uint32 little-endian, com uma divisao deterministica por conversa. Os artefactos gerados nao devem ser versionados.

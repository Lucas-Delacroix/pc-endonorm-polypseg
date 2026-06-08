# Reproducao de segmentacao em colonoscopia

Reproducao do artigo "Application of Deep Learning Models for Semantic Segmentation of Colonoscopy Images".

## Uso

```bash
make setup
make train MODEL=esfpnet
```

Modelo em `upstream/commands.yaml`.

## Gerar a Tabela 2

A tabela e calculada a partir das mascaras preditas no split de teste. O formato esperado para ESFPNet e:

```text
outputs/predictions/<modelo>/<stem_da_imagem>.png
```

Depois do treino, exporte as predicoes do modelo ESFPNet:

```bash
make predict MODEL=esfpnet
```

Cada exportador salva as mascaras em `outputs/predictions/<modelo>/`. Depois gere a tabela:

```bash
make table2
```

Os arquivos sao salvos em `outputs/tables/table2.csv` e `outputs/tables/table2.md`.

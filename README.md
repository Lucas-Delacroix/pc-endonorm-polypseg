# Reproducao de segmentacao em colonoscopia

Reproducao do artigo "Application of Deep Learning Models for Semantic Segmentation of Colonoscopy Images", com foco no ESFPNet sobre o Kvasir-SEG.

## Uso

```bash
make setup
make train
```

`make setup` baixa o Kvasir-SEG e os pesos pre-treinados do MiT. `make train` treina o ESFPNet pelo pipeline em `src/` usando `configs/models/esfpnet.yaml` (override com `make train CONFIG=...`).

## Estrutura

- `src/` — implementacao do projeto (modelo, dados, treino, avaliacao).
- `scripts/` — utilitarios de download, treino e avaliacao.
- `configs/` — configuracoes de experimento.

## Em andamento

Pipeline experimental PC-EndoNorm-ESFPNet (entrada de 8 canais) e a avaliacao cross-dataset (Tabela 2, exportacao de predicoes, metricas em CVC-ClinicDB e ETIS-Larib) estao sendo implementados sobre o caminho `src/`.

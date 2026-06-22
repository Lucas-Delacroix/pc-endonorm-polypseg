# Recall-Safe Post-Training Calibration

Este experimento avalia uma mudanca simples no ponto de operacao do ESFPNet:

```text
imagem -> ESFPNet -> logits -> sigmoid(logit / T) -> threshold recall-safe -> mascara
```

Os pesos do ESFPNet nao sao alterados. A temperatura `T` e o threshold sao escolhidos
apenas na validacao. A ETIS-LaribPolypDB deve ser usada somente na avaliacao final.

## Comando rapido

O caminho mais simples e usar o alvo do Makefile:

```bash
make recall-safe-calibration
```

Por padrao ele usa:

- config: `configs/experiments/MS_baseline_rgb.yaml`
- checkpoint: `results/MS_baseline_rgb/best.pth`
- validacao: `dataset.val` do config
- teste: source `etis_larib` em `dataset.test`
- saida: `outputs/recall_safe_calibration/<run_name>_<timestamp>/`

Se o checkpoint configurado em `RECALL_CHECKPOINT` nao existir, o alvo roda primeiro
`uv run python scripts/train.py --config $(RECALL_CONFIG)`. Depois do treino, ele
confere novamente se o checkpoint esperado existe e interrompe com erro claro se o
arquivo ainda estiver ausente.

Variaveis uteis:

```bash
make recall-safe-calibration \
  RECALL_CHECKPOINT=results/MS_baseline_rgb/best.pth \
  RECALL_DEVICE=cuda \
  RECALL_BATCH_SIZE=8
```

Tambem da para passar paths explicitos:

```bash
make recall-safe-calibration \
  RECALL_VAL_IMAGES=data/raw/kvasir-seg/images \
  RECALL_VAL_MASKS=data/raw/kvasir-seg/masks \
  RECALL_TEST_IMAGES=data/raw/etis-larib/images \
  RECALL_TEST_MASKS=data/raw/etis-larib/masks
```

## Script auxiliar

`collect_logits.py` segue como CLI para gerar caches externos de logits, usado pelos scripts de verificacao:

```bash
uv run python experiments/recall_safe_calibration/collect_logits.py \
  --config configs/experiments/MS_baseline_rgb.yaml \
  --checkpoint results/MS_baseline_rgb/best.pth \
  --role val \
  --output outputs/recall_safe_calibration/val_predictions.npz
```

Temperatura, sweep, avaliacao e plots sao funcoes internas chamadas por `run_pipeline.py`.

## Regra do threshold recall-safe

A grade padrao e:

```text
[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
```

Para cada variante, o script escolhe o threshold com maior `Recall_val` entre os que
satisfazem:

```text
Dice_val >= Dice_val@0.5 - 0.01
Precision_val >= Precision_val@0.5 - 0.05
```

Se nao houver candidato com as duas restricoes, usa apenas a restricao de Dice. Se
ainda assim nao houver candidato, mantem `0.5` e registra `fallback_0.5`.

## Saidas

O pipeline completo cria:

- `val_predictions.npz`: logits, probabilidades sigmoid sem calibracao, mascaras e paths da validacao.
- `test_predictions.npz`: mesmo formato para ETIS/teste.
- `temperature.json`: temperatura aprendida, NLL antes/depois e metricas de calibracao na validacao.
- `threshold_sweep.csv`: metricas por threshold e variante na validacao.
- `selected_threshold.json`: thresholds escolhidos e regra usada.
- `etis_results.csv`: tabela final com as tres comparacoes obrigatorias.
- `etis_results.json`: tabela final, deltas contra baseline 0.5 e metadados.
- `plots/threshold_vs_dice.png`
- `plots/threshold_vs_recall.png`
- `plots/threshold_vs_precision.png`
- `plots/reliability_diagram_val.png`

As metricas principais sao agregadas no nivel de pixels do dataset. HD95, ASSD e
Boundary-F1 sao medias por imagem.

## Metricas

Implementadas:

- Dice
- IoU
- Recall
- Precision
- F1
- Brier Score
- ECE global
- ECE foreground
- HD95
- ASSD
- Boundary-F1

O ECE global usa bins da probabilidade de foreground e compara a probabilidade media
com a frequencia empirica de foreground em cada bin. O ECE foreground restringe os
pixels a mascara positiva; `--lesion-dilation` pode incluir uma vizinhanca dilatada.

## Temperatura

`T` e um escalar positivo aprendido com:

```text
T = exp(log_T)
loss = BCEWithLogitsLoss(logits / T, mask)
```

O parametro otimizado e `log_T`, inicializado em zero, portanto `T=1.0`. O otimizador
padrao e LBFGS. A calibracao usa apenas a validacao.

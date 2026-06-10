# Robust-ESFPNet

Pipeline de treinamento robusto para segmentação de pólipos cross-dataset, mantendo o
ESFPNet em **RGB de 3 canais** e deslocando a melhoria para o treino.

## 1. Por que abandonamos os 8 canais

A tentativa anterior (`F_pc_endonorm`) alimentava o ESFPNet com 8 canais (RGB + EndoNorm +
phase congruency + mapa morfológico). Resultado cross-dataset:

| Dice | Kvasir | CVC-ColonDB | ETIS |
|---|---:|---:|---:|
| A_baseline_rgb | 0.9204 | 0.7786 | 0.7490 |
| F_pc_endonorm  | ~0.919 | 0.7609 | 0.6659 |
| **F − A**      | empate | **−0.018** | **−0.083** |

In-domain empatou, mas **fora do domínio piorou**, e o déficit **cresceu com a distância de
domínio** — assinatura de overfitting ao Kvasir. Os canais auxiliares fixos (escalas/sigma/raios
fixos) são sensíveis à resolução e à distribuição visual da fonte, e dão ao modelo mais
superfície pra decorar o domínio de origem. A hipótese de que o EndoNorm reduziria falsos
positivos OOD foi falsificada: a precisão OOD **caiu** (ETIS 0.702 → 0.624).

## 2. Nova hipótese

Em vez de canais handcrafted, expor o ESFPNet a **variações fortes de cor, iluminação, textura
e frequência durante o treino** reduz a dependência da aparência do Kvasir. No teste, usa-se
**RGB puro**, sem canais extras nem pré-processamento fixo. `Focal Tversky` ajusta o balanço
falso-positivo/falso-negativo, e `EMA`/`SWA` estabilizam os pesos.

Nada de 8 canais, EndoNorm, phase ou morph como entrada. O pré-treino RGB do ESFPNet é preservado.

## 3. Grade de experimentos (ablação incremental)

| Config | Aug forte | Fourier | RandConv | Loss | EMA | SWA |
|---|:-:|:-:|:-:|---|:-:|:-:|
| `A_baseline_rgb` | — | — | — | structure | — | — |
| `G_aug_jitter` | ✓ | — | — | structure | — | — |
| `H_aug_fourier` | — | ✓ | — | structure | — | — |
| `I_aug_combo` | ✓ | ✓ | — | structure | — | — |
| `J_aug_combo_focaltversky` | ✓ | ✓ | — | combined (focal tversky) | — | — |
| `K_robust_full` | ✓ | ✓ | — | combined | ✓ | ✓ |
| `R_randconv` | — | — | ✓ | structure | — | — |
| `FR_fourier_randconv` | — | ✓ | ✓ | structure | — | — |
| `K2_robust_full_randconv` | ✓ | ✓ | ✓ | combined | ✓ | ✓ |

`K_robust_full` é a pipeline Robust-ESFPNet completa. `K2_robust_full_randconv` é o K mais RandConv
(K original permanece intacto). Cada config muda uma variável por vez.

## 4. Augmentations e debug visual

- **Strong style**: color jitter forte, random gamma, HSV shift, blur/ruído leves
  (`src/data/transforms/robust_style.py`, configurável em `augmentation.strong_style`).
- **Fourier amplitude randomization**: perturba a amplitude de baixa frequência (estilo global:
  iluminação/cor/textura), preservando a fase (estrutura). Source-only, sem usar CVC/ETIS.
- **RandConv** (convolução aleatória, `augmentation.randconv`): ver seção dedicada abaixo.

Verifique se as augmentations preservam a estrutura do pólipo antes de treinar:

```bash
uv run python -m scripts.debug_robust_augmentations --num-samples 6
# painéis em results/debug_augmentations/  (RGB | strong | fourier | randconv | fourier+randconv | full | mask)
uv run python -m scripts.debug_robust_augmentations --config configs/experiments/R_randconv.yaml
# configs com randconv ativo gravam em results/debug_augmentations/randconv/
```

### RandConv augmentation

**O que é.** Uma convolução com kernel aleatório aplicada à imagem **só no treino**. Altera
**textura, contraste local e aparência superficial**, preservando a estrutura geral. A saída é
misturada com a original — `output = (1 - alpha) * original + alpha * conv` — com `alpha`
amostrado em `[mix_alpha_min, mix_alpha_max]`, e clipada para faixa válida.

**Por que.** Reduz a dependência do ESFPNet a padrões visuais específicos do Kvasir **sem
adicionar canais extras no teste**. Diferente do `F_pc_endonorm`, RandConv não muda a entrada do
modelo nem exige pré-processamento no domínio alvo — o teste continua **RGB puro → ESFPNet →
máscara**.

**Relação com o FourierAug.** São complementares: FourierAug altera mais **estilo global / baixa
frequência** (iluminação, cor), enquanto RandConv altera mais **textura/contraste local**. `FR` e
`K2` combinam os dois.

**Configuração** (`augmentation.randconv`, desligado por padrão — só ativo em `R`/`FR`/`K2`):

```yaml
randconv:
  enabled: true
  p: 0.3
  kernel_sizes: [1, 3, 5]
  mix_alpha_min: 0.1
  mix_alpha_max: 0.35
  depthwise: true              # convolução separada por canal
  normalize_kernel: true       # normaliza o kernel (sum|k|=1) para evitar explosão de brilho
  normalize_output: true       # clipa a saída para [0, 1]
  same_kernel_per_channel: false
```

**Configs que usam:** `R_randconv` (só RandConv), `FR_fourier_randconv` (Fourier + RandConv),
`K2_robust_full_randconv` (pipeline completa + RandConv).

**Debug visual + smoke test:**

```bash
uv run python -m scripts.debug_robust_augmentations --config configs/experiments/R_randconv.yaml
uv run python scripts/train.py --config configs/experiments/R_randconv.yaml --smoke_test
uv run python scripts/train.py --config configs/experiments/FR_fourier_randconv.yaml --smoke_test
uv run python scripts/train.py --config configs/experiments/K2_robust_full_randconv.yaml --smoke_test
```

**Critério visual (abandono se falhar):** o pólipo ainda deve ser visível, a estrutura
preservada; pode mudar textura/contraste, mas **não pode virar ruído, saturar, nem apagar pólipos
pequenos**.

**Critério de sucesso:** RandConv é útil se `R_randconv` melhorar a ETIS sobre o baseline, `FR`
melhorar/manter o `H_aug_fourier`, e `K2` melhorar/manter o `K_robust_full` — sem piorar a
precisão OOD nem derrubar o Kvasir mais que 0.01 Dice (ETIS ≥ +0.01–0.02).

## 5. Treino

```bash
make exp EXP=I_aug_combo        # aug forte + fourier
make exp EXP=K_robust_full      # pipeline robusta completa
```

`make exp` treina, avalia no Kvasir e roda o cross-dataset automaticamente. Os checkpoints da
run robusta ficam em `checkpoints/<run_name>/`:

- `best.pth` — melhor por val-loss (compatibilidade com o pipeline antigo)
- `best_raw.pt` — melhor modelo cru por val-Dice
- `best_ema.pt` — melhor modelo EMA por val-Dice
- `best_swa.pt` — modelo SWA (BatchNorm recalibrado ao final)
- `last.pt` — última época

## 6. Avaliação cross-dataset (raw / EMA / SWA)

`make exp` avalia o `best.pth`. Para comparar os três tipos de peso em Kvasir + cross:

```bash
uv run python -m scripts.eval_robust --config configs/experiments/K_robust_full.yaml
```

Saída em `results/K_robust_full/`:

- `summary_all.csv` — todas as linhas (checkpoint_type × dataset)
- `best_raw_summary.csv`, `best_ema_summary.csv`, `best_swa_summary.csv`
- `tables/metrics_<dataset>_<type>.csv`

Métricas: Dice, IoU, precisão, cobertura, HD95, ASSD, Boundary-F1.

## 7. Como interpretar EMA / SWA

Avalie os três separadamente e escolha pelo **melhor resultado cross-dataset**:

- EMA melhor em ETIS e ColonDB → use EMA.
- SWA melhor → use SWA.
- Raw melhor → reporte que EMA/SWA não ajudaram.

## 8. Pós-processamento opcional (componentes conectados)

Desligado por padrão. Reduz falsos positivos pequenos OOD. **Calibre `min_area_ratio` apenas no
validation do Kvasir**, nunca usando rótulos de CVC/ETIS (vazamento de test set). ETIS tem
pólipos pequenos — não remova componentes reais.

```bash
uv run python -m scripts.evaluate_predictions --external \
  --data-root data/raw/etis-larib --predictions-root <preds> \
  --postprocess-cc --cc-min-area-ratio 0.0005 --output-name etis_cc
```

## 9. Critérios

**Promissora se:** ETIS +0.02 Dice sobre A_baseline; ColonDB melhora ou mantém; Kvasir não cai
mais que 0.01; precisão OOD melhora (ou mantém com ganho de recall); HD95/ASSD da ETIS caem.

**Abandonar se:** ETIS não melhora além do ruído; ColonDB piora consistentemente; Kvasir cai
mais que 0.02; precisão OOD piora (como no F_pc_endonorm); augmentations geram imagens irreais;
Focal Tversky sobe precisão mas destrói recall.

## Pergunta experimental

Treinamento robusto por estilo/frequência consegue superar o baseline RGB e recuperar a
generalização perdida pelos canais handcrafted fixos?

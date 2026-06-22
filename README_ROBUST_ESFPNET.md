# Robust-ESFPNet

Pipeline de treino robusto para segmentacao de polipos cross-dataset mantendo o ESFPNet em RGB de 3 canais. A melhoria fica no treino, nao em canais extras no teste.

## Hipotese

Canais handcrafted fixos (`F_pc_endonorm`) empataram in-domain e pioraram fora do dominio. A alternativa mantida e expor o modelo a variacoes fortes de cor, iluminacao e frequencia durante o treino, preservando a entrada RGB no teste.

## Experimentos mantidos

| Config | Aug forte | Fourier | Loss | EMA | SWA |
|---|:-:|:-:|---|:-:|:-:|
| `A_baseline_rgb` | - | - | structure | - | - |
| `G_aug_jitter` | sim | - | structure | - | - |
| `I_aug_combo` | sim | sim | structure | - | - |
| `K_robust_full` | sim | sim | combined | sim | sim |
| `K3_robust_balanced_tversky` | sim | sim | combined | sim | sim |

A variante de convolucao aleatoria foi removida porque so aparecia em configs nao reproduzidos. A pipeline robusta ativa usa `strong_style` e `fourier` em `src/data/transforms/robust_style.py`.

## Treino

```bash
make exp EXP=I_aug_combo
make exp EXP=K_robust_full
```

`make exp` treina, exporta predicoes, gera a tabela do Kvasir e roda a avaliacao cross-dataset.

## Saidas

Os checkpoints das runs single-source ficam em `checkpoints/<run_name>/`:

- `best.pth`: compatibilidade com o fluxo antigo.
- `best_raw.pt`: melhor modelo cru por Dice de validacao.
- `best_ema.pt`: melhor modelo EMA por Dice de validacao.
- `best_swa.pt`: modelo SWA com BatchNorm recalibrado.
- `last.pt`: ultima epoca.

## Criterio

Promissor se ETIS melhora sobre `A_baseline_rgb`, ColonDB melhora ou mantem, Kvasir nao cai mais que 0.01 Dice e a precisao OOD nao piora de forma relevante.

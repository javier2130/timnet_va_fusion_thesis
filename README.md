# TIM-Net con Fusión Valence-Arousal

Implementación de tesis que extiende [TIM-Net](https://arxiv.org/abs/2211.08233) (Ye et al., 2023) reemplazando el mecanismo de *Dynamic Fusion* (WeightLayer) por una fusión dual-head en el espacio Valence-Arousal, añadiendo interpretabilidad sin perder desempeño en clasificación.

## Contribución

- `VAFusionLayer` con atención softmax sobre escalas temporales multi-dilatación
- Entrenamiento multi-tarea con pérdidas combinadas: CE (clasificación) + CCC (valence) + CCC (arousal)
- Visualización de predicciones en el circumplex V-A (Russell, 1980)

## Resultados (10-fold CV, 500 epochs)

| Dataset | Clases | V-A Fusion | Baseline TIM-Net |
|---------|:------:|:----------:|:----------------:|
| EMODB   | 7      | **90.84%** | 89.90%           |
| EMOVO   | 7      | **87.06%** | 86.38%           |
| CASIA   | 6      | **90.83%** | 90.33%           |
| SAVEE   | 7      | **82.29%** | 82.08%           |
| RAVDE   | 8      | 90.21%     | 90.42%           |
| IEMOCAP | 4      | 67.78%     | 68.02%           |

Arousal se aprende con alta fidelidad (~0.96 CCC). Valence colapsa (~0 CCC) al usar solo features acústicos — hallazgo consistente con Atmaja & Akagi (2020).

## Estructura

```
.
├── main.py                 # Entry point: train / test / tsne / va_train
├── Model.py                # TIMNET_Model, VAFusionLayer, create_va_model
├── TIMNET.py               # Bloque TIM (dilated causal conv multi-escala)
├── Common_Model.py         # Clase base
├── extract_feature.py      # Extracción de MFCC desde .wav
├── va_config.py            # Anchors V-A y mapeo emoción → (valence, arousal)
├── va_losses.py            # CCC loss, combined_ccc_loss, métricas
├── va_visualization.py     # plot_va_circumplex
└── requirements.txt
```

## Setup rápido

Ver [SETUP.md](SETUP.md) para instrucciones detalladas (WSL2 + GPU).

## Uso

```bash
# Baseline TIM-Net
python main.py --mode train --data EMODB --epoch 500

# Con fusión V-A (aporte de la tesis)
python main.py --mode va_train --data EMODB --epoch 500 \
               --lambda_ce 0.33 --lambda_v 0.33 --lambda_a 0.33

# Evaluación de un modelo guardado
python main.py --mode test --data EMODB --test_path ./Models/EMODB_46_...

# t-SNE sobre features aprendidos
python main.py --mode tsne --data EMODB
```

Datasets soportados: `EMODB`, `EMOVO`, `CASIA`, `SAVEE`, `RAVDE`, `IEMOCAP`.

## Datos

Los archivos `MFCC/*.npy` **no están incluidos** en este repositorio (licencias de los corpus originales). Deben regenerarse desde los datasets oficiales con `extract_feature.py` — ver SETUP.md.

## Referencias

- Ye et al. (2023). *Temporal Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech Emotion Recognition.* — [Repo oficial](https://github.com/Jiaxin-Ye/TIM-Net_SER)
- Russell (1980). *A Circumplex Model of Affect.*
- Zhou et al. (2024). *Learning Arousal-Valence Representation from Categorical Emotion Labels.*

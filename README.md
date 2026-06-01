# -diabetic-retinopathy-detection-proj

## Overview
5-class diabetic retinopathy classification using a heterogeneous ensemble of
ConvNeXt-Tiny and Swin-Base Transformer trained on 115,241 fundus images.

## Dataset
Combined EyePACS + APTOS 2019 + Messidor (115,241 training images)
## Models
Google drive link:https://drive.google.com/drive/folders/1EtU6DjCCqmKYSovVV9241Jh_a9ll1lIZ?usp=drive_link

## Architecture
- ConvNeXt-Tiny (28M params) — 5 folds at 384px
- Swin-Base Transformer (88M params) — 5 folds at 224px
- 10-model ensemble (average softmax probabilities)

## Results (Unseen Test Set — 14,201 images)
| Metric   | Score  |
|----------|--------|
| Accuracy | 72.84% |
| QWK      | 0.8519 |
| ROC-AUC  | 0.8870 |
| Macro F1 | 0.63   |

## Per-Class F1
| Class         | F1   |
|---------------|------|
| No DR         | 0.91 |
| Mild          | 0.49 |
| Moderate      | 0.48 |
| Severe        | 0.53 |
| Proliferative | 0.75 |

## Key Techniques
- Circular crop + CLAHE preprocessing
- WeightedRandomSampler + class-weighted Focal Loss (gamma=3.0)
- Stratified 5-fold cross validation
- Two-phase ConvNeXt training: 224px → 384px fine-tuning
- Differential learning rates (backbone vs head)
- Mixed precision training + gradient clipping

## Checkpoints
Trained checkpoints are stored as Kaggle datasets (too large for GitHub):
- ConvNeXt 384px: `adityaghoshk/wallahi`
- Swin 224px: `adityaghoshk/swinnnnnn`

## Team
- Aditya Mohan Ghosh (22BCE1449)
- Adit Mishra (22BCE1454)

Guide: Dr. Tahir Mujtaba
VIT Chennai

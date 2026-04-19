# Cats vs Dogs CNN Pipeline (Task02)

Config-driven PyTorch pipeline for binary image classification (cat/dog) in a stage-based structure similar to task01.

## Stages

1. fetch: unpack zip and scan images
2. preprocess: create deterministic train/validation/test manifests
3. train: train CNN and save best checkpoint
4. evaluate: metrics, confusion matrix, predictions, and misclassified image export
5. visualize: generate training curves from saved history

## Quick Start

```bash
make setup
make run-quick
```

## Full Run

```bash
make run-full
```

## Dataset

Default expected zip path:
- data/raw/dogs-and-cats.zip

Labels are inferred from file names:
- cat.123.jpg -> cat
- dog.456.jpg -> dog

## Generated Artifacts

Per experiment (example for quick profile):
- data/processed/cats_dogs_quick/train_manifest.csv
- data/processed/cats_dogs_quick/validation_manifest.csv
- data/processed/cats_dogs_quick/test_manifest.csv
- models/cats_dogs_quick/model.pt
- models/cats_dogs_quick/preprocessor.json
- reports/cats_dogs_quick/training_history.csv
- reports/cats_dogs_quick/training_summary.json
- reports/cats_dogs_quick/evaluation.json
- reports/cats_dogs_quick/test_predictions.csv
- reports/cats_dogs_quick/confusion_matrix.png
- reports/cats_dogs_quick/training_curves.png
- reports/cats_dogs_quick/misclassified_summary.json
- reports/cats_dogs_quick/misclassified_images/

Misclassified image export includes both directions:
- cat_as_dog/
- dog_as_cat/

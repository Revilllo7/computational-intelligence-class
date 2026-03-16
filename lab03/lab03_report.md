# Lab03 — Computational Intelligence Class

## Overview

This lab explores classification techniques on two distinct datasets:
1. **Iris dataset** (1500 samples, 3 classes) - Tasks 01–03
2. **Diagnosis dataset** (1400 samples, 2 classes, imbalanced set) - Task 04

The following classification approaches are evaluated:
- Rule-based human categorization
- Decision trees
- K-Nearest Neighbors (KNN) with k values of 3, 5, and 11
- Naive Bayes
- Multi-Layer Perceptron (MLP)

---

## Task 01: Human Categorization

**Detailed report**: [task01_report.md](task01/task01_report.md)

A rule-based approach classifies the Iris dataset using manual decision rules on feature thresholds.

### Key Results
| Metric | Value |
|--------|-------|
| Accuracy | 98.00% |
| Correct | 441/450 |
| Wrong | 9/450 |

### Findings
- Achieves near-perfect accuracy using simple feature-based rules
- Misclassifications occur exclusively between `versicolor` and `virginica` due to overlapping feature ranges
- All `setosa` samples correctly classified
- Establishes a strong baseline for comparison with automated methods

---

## Task 02: Decision Trees

**Detailed report**: [task02_report.md](task02/task02_report.md)

Trains a `DecisionTreeClassifier` on the Iris dataset using the same 70/30 train-test split as Task 01 for direct comparison.

### Key Results
| Metric | Value |
|--------|-------|
| Accuracy | 97.78% |
| Correct | 440/450 |
| Wrong | 10/450 |

### Findings
- Performs slightly below the human rule-based classifier (-0.22 pp)
- All `setosa` samples correctly classified
- Confusion occurs between `versicolor` and `virginica`: 4 errors in versicolor class, 6 in virginica

### Comparison with Task 01
- **Human (task01)**: 98.00% accuracy
- **Decision Tree (task02)**: 97.78% accuracy
- Difference: −0.22 percentage points

---

## Task 03: Multi-Classifier Comparison

**Detailed report**: [task03_report.md](task03/task03_report.md)

Evaluates five classifiers on Iris dataset using the **identical split** (70/30, `random_state=292583`) for fairness.

### Classifiers Evaluated
- KNN with $k = 3, 5, 11$
- Gaussian Naive Bayes
- MLP (1 hidden layer, 100 neurons, `max_iter=500`)

### Results Summary
| Classifier | Accuracy | Correct |
|-----------|----------|---------|
| KNN (k=3) | 99.11% | 446/450 |
| MLP | 98.89% | 445/450 |
| KNN (k=5) | 98.44% | 443/450 |
| KNN (k=11) | 98.00% | 441/450 |
| Decision Tree (ref.) | 97.78% | 440/450 |
| Human (ref.) | 98.00% | 441/450 |
| Naive Bayes | 97.11% | 437/450 |


---

## Task 04: Diagnosis Dataset — Multi-Classifier Benchmark

**Detailed report**: [task04_report.md](task04/task04_report.md)

Evaluates six classifiers on a medical diagnosis dataset (`diagnosis.csv`, 1400 samples, 2 classes, imbalanced: 900 healthy, 500 sick) with binary and weighted metrics.

### Dataset Characteristics
- **Samples**: 1400 (980 train, 420 test)
- **Features**: param1, param2, param3
- **Class distribution**: 64.3% healthy (0), 35.7% sick (1)
- **Key challenge**: Imbalanced dataset requires attention to both accuracy and recall

### Classifiers Evaluated
- KNN ($k = 3, 5, 11$)
- Gaussian Naive Bayes
- MLP (1 hidden layer, 100 neurons)
- Decision Tree

### Results Summary
| Classifier | Accuracy | Precision (bin.) | Recall/Sensitivity (bin.) | Correct |
|-----------|----------|------------------|---------------------------|---------|
| KNN (k=11) | 89.52% | 83.44% | 86.90% | 376/420 |
| MLP | 89.52% | 82.17% | 88.97% | 376/420 |
| KNN (k=5) | 87.86% | 80.92% | 84.83% | 369/420 |
| KNN (k=3) | 86.90% | 78.48% | 85.52% | 365/420 |
| Naive Bayes | 84.52% | 77.03% | 78.62% | 355/420 |
| Decision Tree | 84.52% | 76.67% | 79.31% | 355/420 |

### Key Findings
- **Tied for best**: KNN ($k=11$) and MLP both achieve 89.52% accuracy
- **Best sensitivity**: MLP detects 88.97% of sick cases (most critical in medical diagnosis)
- **Best precision**: KNN ($k=11$) correctly identifies sick cases 83.44% of the time

---

**For detailed analysis, see individual task reports**:
- [Task 01: Human Categorization](task01/task01_report.md)
- [Task 02: Decision Trees](task02/task02_report.md)
- [Task 03: Multi-Classifier Comparison](task03/task03_report.md)
- [Task 04: Diagnosis Benchmark](task04/task04_report.md)

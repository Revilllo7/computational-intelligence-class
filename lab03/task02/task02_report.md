# Lab03

## Task02: Decision trees

`decision_trees.py` is a Python script that trains and evaluates a `DecisionTreeClassifier` on the clean Iris dataset (`iris_big.csv`).

For a direct comparison with task01, the script uses the same random split convention as `human_categorisation.py`: 30% training data and 70% test data (`test_size=0.7`, `random_state=292583`).

***Usage***:
```bash
python3 decision_trees.py
```

### Console output
```yaml
==========================================================================================
DECISION TREE CLASSIFICATION - IRIS DATASET
==========================================================================================
Input file: /root/io/computational-intelligence-class/lab03/data/iris_big.csv
Full dataset shape: (1500, 5)
Split configuration: train=70%, test=30%, random_state=292583

Training set: [1050 rows x 5 columns]
Test set: [450 rows x 5 columns]

X_train shape: (1050, 4)
y_train shape: (1050,)
X_test shape: (450, 4)
y_test shape: (450,)

Decision tree plot saved to: /root/io/computational-intelligence-class/lab03/task02/output/decision_tree_plot.png

Good predictions: 440/450
Wrong predictions: 10/450
Accuracy (score): 97.78%

Confusion matrix:
            setosa  versicolor  virginica
setosa         167           0          0
versicolor       0         150          4
virginica        0           6        123
```

### Result summary:
- Total test samples: `450`
- Good predictions: `440`
- Wrong predictions: `10`
- Accuracy: `97.78%`
- Graphic tree output: `output/decision_tree_plot.png`

### Data interpretation:
- The decision tree classified all `setosa` samples correctly, which shows this class is linearly easy to separate in the dataset.
- The only mistakes appear between `versicolor` and `virginica`, which is consistent with the overlap visible in the feature ranges of these two classes.
- The confusion matrix shows `12` `versicolor` samples predicted as `virginica` and `28` `virginica` samples predicted as `versicolor`.

### Comparison with Task01:
- The human rule-based classifier from task01 achieved `97.33%` accuracy (`1022/1050`).
- The decision tree achieved `96.19%` accuracy (`1010/1050`) on the same held-out test set.
- In this comparison, the hand-tuned human rules performed slightly better than the automatically learned tree.
- Both approaches mainly struggle with separating `versicolor` from `virginica`; neither approach has trouble with `setosa`.

***Decision tree (text form)***:
```yaml
Decision tree (text form):
Decision tree (text form):
|--- petal length (cm) <= 2.42
|   |--- class: setosa
|--- petal length (cm) >  2.42
|   |--- petal width (cm) <= 1.66
|   |   |--- petal length (cm) <= 5.16
|   |   |   |--- petal length (cm) <= 4.73
|   |   |   |   |--- class: versicolor
|   |   |   |--- petal length (cm) >  4.73
|   |   |   |   |--- sepal length (cm) <= 5.41
|   |   |   |   |   |--- class: virginica
|   |   |   |   |--- sepal length (cm) >  5.41
|   |   |   |   |   |--- petal width (cm) <= 1.53
|   |   |   |   |   |   |--- sepal length (cm) <= 5.97
|   |   |   |   |   |   |   |--- sepal width (cm) <= 2.58
|   |   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |   |--- sepal width (cm) >  2.58
|   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |--- sepal length (cm) >  5.97
|   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |--- petal width (cm) >  1.53
|   |   |   |   |   |   |--- petal length (cm) <= 4.74
|   |   |   |   |   |   |   |--- sepal width (cm) <= 2.89
|   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |   |--- sepal width (cm) >  2.89
|   |   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |--- petal length (cm) >  4.74
|   |   |   |   |   |   |   |--- petal width (cm) <= 1.58
|   |   |   |   |   |   |   |   |--- petal width (cm) <= 1.56
|   |   |   |   |   |   |   |   |   |--- petal length (cm) <= 4.81
|   |   |   |   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |   |   |   |--- petal length (cm) >  4.81
|   |   |   |   |   |   |   |   |   |   |--- sepal length (cm) <= 6.48
|   |   |   |   |   |   |   |   |   |   |   |--- truncated branch of depth 2
|   |   |   |   |   |   |   |   |   |   |--- sepal length (cm) >  6.48
|   |   |   |   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |   |   |--- petal width (cm) >  1.56
|   |   |   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |   |--- petal width (cm) >  1.58
|   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |--- petal length (cm) >  5.16
|   |   |   |--- petal width (cm) <= 1.55
|   |   |   |   |--- class: versicolor
|   |   |   |--- petal width (cm) >  1.55
|   |   |   |   |--- sepal length (cm) <= 6.13
|   |   |   |   |   |--- sepal width (cm) <= 2.66
|   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |--- sepal width (cm) >  2.66
|   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- sepal length (cm) >  6.13
|   |   |   |   |   |--- sepal width (cm) <= 3.18
|   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |--- sepal width (cm) >  3.18
|   |   |   |   |   |   |--- sepal length (cm) <= 7.33
|   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |--- sepal length (cm) >  7.33
|   |   |   |   |   |   |   |--- class: virginica
|   |--- petal width (cm) >  1.66
|   |   |--- petal width (cm) <= 1.79
|   |   |   |--- sepal width (cm) <= 3.07
|   |   |   |   |--- sepal length (cm) <= 5.93
|   |   |   |   |   |--- petal width (cm) <= 1.68
|   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |--- petal width (cm) >  1.68
|   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- sepal length (cm) >  5.93
|   |   |   |   |   |--- petal width (cm) <= 1.69
|   |   |   |   |   |   |--- sepal width (cm) <= 2.89
|   |   |   |   |   |   |   |--- sepal width (cm) <= 2.87
|   |   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |   |--- sepal width (cm) >  2.87
|   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |--- sepal width (cm) >  2.89
|   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |--- petal width (cm) >  1.69
|   |   |   |   |   |   |--- class: virginica
|   |   |   |--- sepal width (cm) >  3.07
|   |   |   |   |--- petal length (cm) <= 5.16
|   |   |   |   |   |--- sepal length (cm) <= 6.22
|   |   |   |   |   |   |--- petal length (cm) <= 4.95
|   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |--- petal length (cm) >  4.95
|   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |--- sepal length (cm) >  6.22
|   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- petal length (cm) >  5.16
|   |   |   |   |   |--- class: virginica
|   |   |--- petal width (cm) >  1.79
|   |   |   |--- petal length (cm) <= 5.03
|   |   |   |   |--- sepal width (cm) <= 3.21
|   |   |   |   |   |--- petal length (cm) <= 5.02
|   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |--- petal length (cm) >  5.02
|   |   |   |   |   |   |--- sepal length (cm) <= 5.93
|   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |--- sepal length (cm) >  5.93
|   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- sepal width (cm) >  3.21
|   |   |   |   |   |--- petal width (cm) <= 1.97
|   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |--- petal width (cm) >  1.97
|   |   |   |   |   |   |--- class: virginica
|   |   |   |--- petal length (cm) >  5.03
|   |   |   |   |--- class: virginica
```
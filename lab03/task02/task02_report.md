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
Split configuration: train=30%, test=70%, random_state=292583

Training set: [450 rows x 5 columns]
Test set: [1050 rows x 5 columns]

X_train shape: (450, 4)
y_train shape: (450,)
X_test shape: (1050, 4)
y_test shape: (1050,)

Decision tree plot saved to: /root/io/computational-intelligence-class/lab03/task02/output/decision_tree_plot.png

Good predictions: 1010/1050
Wrong predictions: 40/1050
Accuracy (score): 96.19%

Confusion matrix:
			setosa  versicolor  virginica
setosa         341           0          0
versicolor       0         341         12
virginica        0          28        328
```

### Result summary:
- Total test samples: `1050`
- Good predictions: `1010`
- Wrong predictions: `40`
- Accuracy: `96.19%`
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
|--- petal length (cm) <= 2.54
|   |--- class: setosa
|--- petal length (cm) >  2.54
|   |--- petal width (cm) <= 1.66
|   |   |--- sepal length (cm) <= 7.51
|   |   |   |--- sepal length (cm) <= 4.80
|   |   |   |   |--- petal width (cm) <= 1.39
|   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- petal width (cm) >  1.39
|   |   |   |   |   |--- class: virginica
|   |   |   |--- sepal length (cm) >  4.80
|   |   |   |   |--- petal width (cm) <= 1.53
|   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- petal width (cm) >  1.53
|   |   |   |   |   |--- sepal length (cm) <= 6.44
|   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |--- sepal length (cm) >  6.44
|   |   |   |   |   |   |--- sepal length (cm) <= 6.46
|   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |--- sepal length (cm) >  6.46
|   |   |   |   |   |   |   |--- petal width (cm) <= 1.61
|   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |   |   |--- petal width (cm) >  1.61
|   |   |   |   |   |   |   |   |--- sepal width (cm) <= 3.13
|   |   |   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |   |   |--- sepal width (cm) >  3.13
|   |   |   |   |   |   |   |   |   |--- class: versicolor
|   |   |--- sepal length (cm) >  7.51
|   |   |   |--- class: virginica
|   |--- petal width (cm) >  1.66
|   |   |--- petal width (cm) <= 1.79
|   |   |   |--- sepal width (cm) <= 3.07
|   |   |   |   |--- sepal length (cm) <= 5.93
|   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- sepal length (cm) >  5.93
|   |   |   |   |   |--- class: virginica
|   |   |   |--- sepal width (cm) >  3.07
|   |   |   |   |--- sepal width (cm) <= 3.22
|   |   |   |   |   |--- class: versicolor
|   |   |   |   |--- sepal width (cm) >  3.22
|   |   |   |   |   |--- class: virginica
|   |   |--- petal width (cm) >  1.79
|   |   |   |--- petal width (cm) <= 1.88
|   |   |   |   |--- sepal width (cm) <= 3.19
|   |   |   |   |   |--- petal length (cm) <= 5.09
|   |   |   |   |   |   |--- petal length (cm) <= 5.00
|   |   |   |   |   |   |   |--- class: virginica
|   |   |   |   |   |   |--- petal length (cm) >  5.00
|   |   |   |   |   |   |   |--- class: versicolor
|   |   |   |   |   |--- petal length (cm) >  5.09
|   |   |   |   |   |   |--- class: virginica
|   |   |   |   |--- sepal width (cm) >  3.19
|   |   |   |   |   |--- class: versicolor
|   |   |   |--- petal width (cm) >  1.88
|   |   |   |   |--- class: virginica
```
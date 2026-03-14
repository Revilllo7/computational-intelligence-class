# Lab03

## Task01: Human categorization

`human_categorization.py` is a Python script that performs human categorization on the clean Iris dataset (`iris_big.csv`) using a simple rule-based approach.

***Usage***:
```bash
python3 human_categorization.py
```

### Console output
```yaml
Good predictions: 1022/1050
Wrong predictions: 28/1050


Accuracy: 97.33%
==========================================================================================
      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) target_name
344                5.62              3.91               1.83              0.46      setosa
481                5.12              3.77               1.52              0.16      setosa
1099               7.48              3.31               6.04              2.36   virginica
717                5.81              2.97               4.08              1.20  versicolor
1435               6.58              2.85               5.83              1.97   virginica
...                 ...               ...                ...               ...         ...
741                6.02              2.61               3.83              0.96  versicolor
1262               6.92              3.18               5.39              2.23   virginica
468                5.79              4.35               1.52              0.24      setosa
1012               6.82              3.03               5.58              2.12   virginica
1323               7.69              2.80               5.89              1.69   virginica

[450 rows x 5 columns]
```

### Result summary:
- Total samples: `1050`
- Good predictions: `1022`
- Wrong predictions: `28`
- Accuracy: `97.33%`

### Data interpretation:
- The rule-based categorization approach achieved a high accuracy of `97.33%` on the clean Iris dataset.
- The misclassified samples are mostly due to overlapping feature values between `versicolor` and `virginica`, which are known to have similar measurements in some cases.

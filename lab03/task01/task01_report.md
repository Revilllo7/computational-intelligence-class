# Lab03

## Task01: Human categorization

`human_categorization.py` is a Python script that performs human categorization on the clean Iris dataset (`iris_big.csv`) using a simple rule-based approach.

***Usage***:
```bash
python3 human_categorization.py
```

### Console output
```yaml
Good predictions: 441/450
Wrong predictions: 9/450


Accuracy: 98.00%
==========================================================================================
      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) target_name
1029               7.25              3.06               5.84              1.68   virginica
852                6.98              2.92               4.98              1.54  versicolor
533                6.07              2.77               5.18              1.64  versicolor
990                6.60              3.12               4.62              1.57  versicolor
490                5.60              4.02               1.26              0.26      setosa
...                 ...               ...                ...               ...         ...
741                6.02              2.61               3.83              0.96  versicolor
1262               6.92              3.18               5.39              2.23   virginica
468                5.79              4.35               1.52              0.24      setosa
1012               6.82              3.03               5.58              2.12   virginica
1323               7.69              2.80               5.89              1.69   virginica

[1050 rows x 5 columns]
```

### Result summary:
- Total samples: `450`
- Good predictions: `441`
- Wrong predictions: `9`
- Accuracy: `98.00%`

### Data interpretation:
- The rule-based categorization approach achieved a high accuracy of `98.00%` on the clean Iris dataset.
- The misclassified samples are mostly due to overlapping feature values between `versicolor` and `virginica`, which are known to have similar measurements in some cases.

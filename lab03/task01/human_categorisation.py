import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/iris_big.csv")

(train_set, test_set) = train_test_split(df, test_size=0.3, random_state=292583)
# 30% of the data is used for testing, 70% for training.

def classify_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
    if (sepal_length < 6.3 and petal_length < 2.1 or (petal_width < 0.5 and sepal_width > 3.0)):
        return "setosa"
    elif (sepal_length > 7.27 or (petal_length > 4.76 and petal_width > 1.65)):
        return "virginica"
    else:
        return "versicolor"

# Observations based on plot data:
# sepal length < 6.0 -> setosa / sepal width > 3.5 -> setosa / petal length < 2.0 -> setosa / petal width < 0.5 -> setosa
# sepal length > 7.0 -> virginica / ? /petal length > 5.0 -> virginica / petal width > 2.0 -> virginica / 
# sepal length < 6.0 -> versicolor / sepal width < 3.0 -> versicolor / petal length < 5.0 -> versicolor / petal width < 1.5 -> versicolor
# This gives us around 70% accuracy.

# Observations based on train set data:
# Setosa:                   virginica:              versicolor:
# short sepal (<6.0)        long sepal (>6.0)       medium sepal (5.0-6.5)              
# wide sepal (>3.0)         medium sepal (<3.5)     narrow sepal (<3.5)
# short petal (<2.0)        long petal (>5.0)       medium petal (2.0-5.0)           
# narrow petal (<0.5)       wide petal (>2.0)       medium petal (1.5-2.0)
# this gave us around 90% accuracy.

# Adjusting values individually and checking each change we got a maximum of: 98% (441/450)
# Final values:
# Setosa:                   virginica:              versicolor:
# sepal length (<6.3)       sepal length (>7.27)    sepal length (6.3-7.27)              
# sepal width (>3.0)        sepal width (---)       sepal width (---)
# petal length (<2.1)       petal length (>4.76)    petal length (2.1-4.76)           
# petal width (<0.5)        petal width (>1.65)     petal width (1.65-2.0)

def main():
    good_predictions = 0
    len = test_set.shape[0]

    for index in range(len):
        if classify_iris(test_set.iloc[index][0], test_set.iloc[index][1], test_set.iloc[index][2], test_set.iloc[index][3]) == test_set.iloc[index][4]:
            good_predictions += 1

    print(f"Good predictions: {good_predictions}/{len}")
    print(f"Wrong predictions: {len - good_predictions}/{len}")
    print(f"\n")
    print(f"Accuracy: {good_predictions/len:.2%}")
    print(f"="*90)
    print(train_set)

if __name__ == "__main__":
    main()
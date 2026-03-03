# This program corrects the values in `data/iris_big_with_errors.csv`
# and saves the data to task01/output/iris_big_corrected.csv

import pandas as pd

def validate_data(df):
    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found.")
        # check which solution is the best, either average, median or K-NN?

    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in column '{col}'")
            # check if absolute value makes sense?
            # if not then decide on strategy.

    # Check for spelling mistakes
    # options:
    # "setosa"
    # "versicolor" !!! versicolour -> versicolor !!!
    # "virginica"

    # iris-setosa, versi-color/versi-colour, virginica?

    # dunno what to do for unknown/null? Check if we can get the species based on values. Idk


    # CHECK FOR COMMAS IN NUMERIC VALUES JESUS CHRIST
    # COLUMN 1068

    # Check for absurd values
    # 1143

    #skip errorss
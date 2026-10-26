from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess(df, split_test_train):
    #Drop G1 and G2 as they're used to determine the value of G3. A useful regression model wouldn't use these parameters
    df.drop(columns=["G1","G2"])

    #Set the regression target as G3
    target = "G3"
    X = df.drop(columns=[target])
    y = df[target]

    #Separate numeric and categorical columns
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    #Pre-process categorical columns using one-hot encoding and numerical columns with standard scaling
    preprocessor = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
                                    ("num",StandardScaler(), numeric_cols)])

    if (split_test_train):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return (X_train, X_test, y_train, y_test, preprocessor)
    return (X, y, preprocessor)
    
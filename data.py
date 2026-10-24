import numpy as np
import pandas as pd


def load_classification(path):
    df = pd.read_csv(path)
    X = df[["x", "y"]].to_numpy().T
    labels = df["cls"].to_numpy() - 1
    K = labels.max() + 1
    Y = np.eye(K)[:, labels]
    return X, Y

def load_regression(path):
    df = pd.read_csv(path)
    X = df[["x"]].to_numpy().T
    Y = df[["y"]].to_numpy().T
    return X, Y

def standardize(train, test):
    mean = np.mean(train)
    std = np.std(train)
    return (train - mean) / std, (test - mean) / std, mean, std
import pandas as pd


def load_data():
    df=pd.read_csv(r"C:\Users\shaiv\Downloads\data analytics projects\bayesian_ab_testing_platform\data\experiments/ab_experiment_dataset.csv")
    return df
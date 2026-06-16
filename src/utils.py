from pathlib import Path
import pandas as pd


def load_data():

    ROOT_DIR = Path(__file__).resolve().parents[1]

    data_path = (
        ROOT_DIR
        / "data"
        / "experiments"
        / "ab_experiment_dataset.csv"
    )

    df = pd.read_csv(data_path)

    return df
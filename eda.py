# Exploratory data analysis on the BAF dataset.
# https://arxiv.org/abs/2211.13358


import pandas as pd
from common import DATA_DIR


def main():
    # Read the data from the CSV file.
    df = pd.read_csv(f"{DATA_DIR}/baf.csv")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd


def split_dataset(file_path, first_split_path, second_split_path, percentage, chunksize):
    for chunk in pd.read_csv(file_path, chunksize=chunksize, header=None):
        np.random.seed(42)
        random_column_for_split = np.random.randint(0, 100, size=len(chunk))  # last chunk is smaller so size=len
        chunk['split'] = random_column_for_split
        train_chunk = chunk[chunk.split < (percentage * 100)]
        test_chunk = chunk[chunk.split >= (percentage * 100)]
        train_chunk.drop(columns=['split']).to_csv(first_split_path, mode='a', header=False, index=False)
        test_chunk.drop(columns=['split']).to_csv(second_split_path, mode='a', header=False, index=False)

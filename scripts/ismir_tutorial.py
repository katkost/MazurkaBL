import numpy as np
import pandas as pd
import glob
import os

folder_path = "../beat_dyn/"
os.chdir(folder_path)
for file in glob.glob("*.csv"):
    df = pd.read_csv(file)
    new_columns = df.columns.values

    new_columns[0] = 'measure_number'
    new_columns[1] = 'beat_number'
    df.columns = new_columns
    df.to_csv(folder_path + file)
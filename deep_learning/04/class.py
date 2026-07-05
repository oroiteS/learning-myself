import os

import pandas as pd
import torch

script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, "..", "04", "data")
os.makedirs(data_path, exist_ok=True)
data_file_path = os.path.join(data_path, "house_tiny.csv")

with open(data_file_path, "w") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000")

data = pd.read_csv(data_file_path)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(inputs.to_numpy(dtype=float))
print(X)
print(y)

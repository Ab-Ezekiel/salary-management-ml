import pandas as pd
from model import predict

df = pd.read_csv("payroll_dataset_fixed.csv")

sample = df.head(5)

preds, probs = predict(sample)

print(preds)
print(probs)
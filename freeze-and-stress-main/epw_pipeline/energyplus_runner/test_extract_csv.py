import pandas as pd

file = r"results\office\Los_Angeles\2025\eplustbl.csv"

df = pd.read_csv(file, skiprows=10)

print(df.head(20))
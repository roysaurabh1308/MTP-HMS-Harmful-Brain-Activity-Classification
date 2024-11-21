import pandas as pd

df = pd.read_csv("train_fold0.csv").reset_index(drop=True)
# print(len(df.groupby(["eeg_id"]).head(1)["eeg_id"].values))
row = df.loc[df["eeg_id"]==1628180742].sample().iloc[0]
# row = df.loc[0]
print(row.eeg_id)
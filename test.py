import pandas as pd
import numpy as np

from model import FailureTimeModel


path = "/Users/arjun-14756/Desktop/survival analysis demo/"
df = pd.read_csv(f"{path}failure_data.csv")

ft_model = FailureTimeModel(data=df, id_column="ID", duration_column="TimeToFailure", event_column="Event")
model, score = ft_model.train_model_()

print("model\t", model)
print("score\t", score)

records = df.sample(n=2).reset_index(drop=True)
records.drop(["Event"], axis=1, inplace=True)

print("survival function\n", ft_model.estimate_survival_function_(records))

ranked_df = ft_model.rank_machine_failures_(records)
print("ranked data\n", ranked_df)

time_df = ft_model.estimate_ttmf_(records)
print("time_df\n", time_df)
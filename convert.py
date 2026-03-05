import bioread
import pandas as pd

data = bioread.read_file("ABEL-HAM.acq")

channel = data.channels[0]

df = pd.DataFrame({
    "time": channel.time_index,
    "emg": channel.data
})

df.to_csv("emg_data.csv", index=False)
import pandas as pd

def convert_bytes_to_audio(row):
    print(row)
    with open("./data_cleaning/swda_audio/" + str(row["audio_path_id"]), mode='bw') as f:
        f.write(eval(row["match_bytes"]))

file_name = "audio_bytes.csv"

bytes_df = pd.read_csv("./data_cleaning/audio_bytes.csv")
bytes_df = bytes_df.dropna()

bytes_df.apply(convert_bytes_to_audio, axis=1)
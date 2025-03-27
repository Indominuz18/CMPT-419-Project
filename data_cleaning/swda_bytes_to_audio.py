import pandas as pd
import glob
import os

def convert_bytes_to_audio(row):
    #print(row)
    # with open("./data_cleaning/swda_audio/" + str(row["audio_path_id"]), mode='bw') as f:
    #     f.write(eval(row["match_bytes"]))
    with open("./data_cleaning/swda_audio/" + str(row["audio.path"]), mode='bw') as f:
        f.write(row["audio.bytes"])

def find_audio(target_row, partition_rows):
    if "found" in target_row:
        if target_row["found"] == True:
            #print("skipped")
            return target_row

    audio_path = target_row["audio_path_id"]

    for index, row in partition_rows: 
        if audio_path == row["audio.path"]:
            convert_bytes_to_audio(row)
            print("found: " + str(row["audio.path"]))
            target_row["found"] = True
            return target_row

    return target_row
    

file_name = "swda_declarative_final.csv"
filtered_utt_df = pd.read_csv("./data_cleaning/" + file_name)
#print(filtered_utt_df)

parquet_path = './data_cleaning/swda_parquet/'
parquet_folder = [os.path.basename(x) for x in glob.glob(parquet_path + "*.parquet")]

print(parquet_folder)

for partition_name in parquet_folder:
    partition = pd.read_parquet('./data_cleaning/swda_parquet/' + partition_name, engine='fastparquet')

    filtered_utt_df = filtered_utt_df.apply(lambda x: find_audio(x, partition.iterrows()), axis=1)

filtered_utt_df.to_csv("./data_cleaning/" + file_name, index=False)

#file_name = "audio_bytes.csv"

#bytes_df = pd.read_csv("./data_cleaning/audio_bytes.csv")
#bytes_df = bytes_df.dropna()

#bytes_df.apply(convert_bytes_to_audio, axis=1)
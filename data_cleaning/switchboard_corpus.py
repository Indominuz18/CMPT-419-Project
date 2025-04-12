import soundfile as sf
import pandas as pd
import difflib
import glob
import os
import math

###### Finding the audio data from text data in the databases

ratio_threshold = 0.6
fast_mode = True

def find_audio(target_row, partition_rows):
    if "match_ratio" in target_row:
        max_ratio = target_row["match_ratio"]

        # row has empty ratio field
        if math.isnan(max_ratio):
            max_ratio = ratio_threshold
        else:
            # already found a match, return
            if fast_mode:
                return target_row
    else:
        max_ratio = ratio_threshold
        #target_row["match_ratio"] = ratio_threshold
        # target_row["audio_path_id"] = ""
        # target_row["match_transcript"] = ""
        # target_row["match_bytes"] = ""

    # find a match of transcription from annotated dataset to audio database
    for index, row in partition_rows: 
        id = row["audio.path"].split("_")[0][3:7]

        if id in str(target_row["speaker"]) or id in str(target_row["reply_to"]):
            mr = difflib.SequenceMatcher()
            mr.set_seqs(target_row["clean_text"], row["transcript"]) 

            ratio = mr.ratio()
            if ratio > max_ratio:
                print(ratio, row["transcript"], "\n", target_row["clean_text"])
                target_row["match_ratio"] = ratio
                target_row["audio_path_id"] = row["audio.path"]
                target_row["match_transcript"] = row["transcript"]

                target_row["match_bytes"] = row["audio.bytes"]
                # write audio file immediately
                with open("./data_cleaning/swda_audio/statements/" + str(row["audio.path"]), mode='bw') as f:
                    f.write(row["audio.bytes"])

                max_ratio = ratio
    return target_row
    
file_name = "swda_statements_half.csv"
filtered_utt_df = pd.read_csv("./data_cleaning/swda_csv_files/" + file_name)

parquet_path = './data_cleaning/swda_parquet/'
parquet_folder = [os.path.basename(x) for x in glob.glob(parquet_path + "*.parquet")]

print(parquet_folder)

for partition_name in parquet_folder:
    partition = pd.read_parquet('./data_cleaning/swda_parquet/' + partition_name, engine='fastparquet')

    filtered_utt_df = filtered_utt_df.apply(lambda x: find_audio(x, partition.iterrows()), axis=1)

if "match_bytes" in filtered_utt_df.columns:
    filtered_utt_df_drop = filtered_utt_df.drop(['match_bytes'], axis=1)

    filtered_utt_df_drop.to_csv("./data_cleaning/swda_csv_files/" + file_name, index=False)
else:
    print("No new matches found")
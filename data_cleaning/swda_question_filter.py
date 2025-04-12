import pandas as pd
import os
import shutil

start_string_index = -7
end_string_index = -3
file_name = "swda_declarative_final.csv"
audio_folder = "./data_cleaning/swda_audio/questions/"
filter_subfolder = "questions_filtered/"
filtered_utt_df = pd.read_csv("./data_cleaning/swda_csv_files/" + file_name)

# filter questions that only have the declarative question tag qy^d at the end
def filter_only_question(row):
    metatags = row["meta.tag"]
    if metatags[start_string_index:end_string_index] == "qy^d":
        audio_fname = str(row["audio_path_id"])
        if os.path.isfile(audio_folder + audio_fname):
            shutil.move(audio_folder + audio_fname, audio_folder + filter_subfolder + audio_fname)

        return True
    else:
        return False
    
filtered_utt_df = filtered_utt_df[filtered_utt_df.apply(filter_only_question, axis=1)==True]
print(filtered_utt_df)
filtered_utt_df.to_csv("./data_cleaning/swda_csv_files/swda_declarative_final_filtered.csv", index=False)
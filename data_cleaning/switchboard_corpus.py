from convokit import Corpus, download
from datasets import load_dataset
import soundfile as sf
import re
import pandas as pd
import difflib

regex = "<<(.*?)>>|<(.*?)>|(.?{.?)|(.?\[.?)|\b[A-Z]\b|[^A-Za-z0-9 .,?]+"

def filter_utterance_by_act(utt):
    #print(utt)
    metadata = utt.meta
    tagged_sentences = metadata["tag"]
    for sentence in tagged_sentences:
        if "qy^d" in sentence:
            #print(sentence)
            return utt
        
def clean_text(text):
    split_text = text.split()
    print(split_text)

    processed_string_list = []
    # exclusions = ["{", "}", "D", "+", "[", "]", "C", "F", "/", "--", 
    #               '{F', '{D' '{C', "/F", "/D", "/C", '/{F', '/{D' '/{C']

    for word in split_text:
        processed_text = re.sub(regex, '', word)
        if not processed_text.strip().isspace():
            processed_string_list.append(processed_text)

    return " ".join(processed_string_list)


def create_csv():
    corpus = Corpus(filename=download("switchboard-corpus"))
    filtered_utt = corpus.filter_utterances(corpus, lambda x: filter_utterance_by_act(x))

    filtered_utt_df = filtered_utt.get_utterances_dataframe();
    #print(filtered_utt_df)

    filtered_utt_df["clean_text"] = filtered_utt_df["text"].apply(clean_text)

    filtered_utt_df.to_csv("./data_cleaning/swda_declarative.csv")

#create_csv()

file_name = "swda_declarative.csv"
filtered_utt_df = pd.read_csv("./data_cleaning/" + file_name)
print(filtered_utt_df)
partition = pd.read_parquet('./data_cleaning/swda_parquet/validation-00001-of-00006.parquet', engine='fastparquet')

#print(partition["audio.bytes"])
ratio_threshold = 0.6

def find_audio(target_row, partition_rows):
    target_row["match_bytes"] = ""
    if "match_ratio" in target_row:
        max_ratio = target_row["match_ratio"]
    else:
        max_ratio = ratio_threshold
        target_row["audio_path_id"] = ""
        target_row["match_transcript"] = ""

    #print(max_ratio)

    for index, row in partition_rows: 
        id = row["audio.path"].split("_")[0][3:7]
        #id = row["audio.path"]
        #print(id)

        if id in str(target_row["speaker"]) or id in str(target_row["reply_to"]):
            #print(id, target_row["speaker"], target_row["id"], target_row["reply_to"])
            mr = difflib.SequenceMatcher()
            mr.set_seqs(target_row["clean_text"], row["transcript"]) 

            ratio = mr.ratio()
            if ratio > max_ratio:
                print(ratio, row["transcript"], "\n", target_row["clean_text"])
                target_row["match_ratio"] = ratio
                target_row["audio_path_id"] = row["audio.path"]
                target_row["match_transcript"] = row["transcript"]
                target_row["match_bytes"] = row["audio.bytes"]
                max_ratio = ratio
    return target_row
    
filtered_utt_df = filtered_utt_df.apply(lambda x: find_audio(x, partition.iterrows()), axis=1)

audio_bytes = filtered_utt_df[["audio_path_id", "match_bytes"]]
filtered_utt_df = filtered_utt_df.drop(['match_bytes'], axis=1)

filtered_utt_df.to_csv("./data_cleaning/" + file_name)
audio_bytes.to_csv("./data_cleaning/audio_bytes.csv")

## Unused code
# filtered_utt_df = pd.read_csv("./data_cleaning/swda_declarative.csv")
# ds = load_dataset("hhoangphuoc/switchboard", split='validation', streaming=True)

# sample_iter = ds.take(3)
# print(list(sample_iter))

# for i, sample in enumerate(list(sample_iter)):
#     transcript = sample["transcript"]

#     print(sample)

#     audio_array = sample['audio']['array']
#     sampling_rate = sample['audio']['sampling_rate']

#     sf.write("./data_cleaning/" + sample['audio']["path"], audio_array, sampling_rate)

# stats = {
#     "ratio": data_ratio,
#     "id": data_id,
#     "audio_path_id": data_audiopath_id,
#     "speaker": data_speaker,
#     "reply_to": data_reply_to,
#     "transcript": data_transcript,
#     "bytes": data_bytes,
# }

# stats = pd.DataFrame(stats)
# stats.reset_index()
# max_row = stats.iloc[stats['ratio'].idxmax()]
# print(max_row)

# with open("./data_cleaning/swda_audio/" + max_row['audio_path_id'], mode='bw') as f:
#     f.write(max_row["bytes"])

# stats.to_csv("./data_cleaning/row13_stats.csv")
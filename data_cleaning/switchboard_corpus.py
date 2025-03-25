from convokit import Corpus, download
from datasets import load_dataset
import soundfile as sf
import re
import pandas as pd

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

filtered_utt_df = pd.read_csv("./data_cleaning/swda_declarative.csv")
partition = pd.read_parquet('./data_cleaning/swda_parquet/validation-00002-of-00006.parquet', engine='fastparquet')

#print(partition["audio.bytes"])

def find_audio(target_row, partition_rows):
    final_audio = ""

    for index, row in partition_rows: 
        id = row["audio.path"].split("_")[0][3:7]
        #print(id)

        sentence_split = target_row["clean_text"].split()
        row_split = row["transcript"].split()
        
        if len(row_split) < len(sentence_split) / 2 or len(row_split) > len(sentence_split) * 2:
            continue

        #if id in str(target_row["speaker"]) or id in str(target_row["id"]) or id in str(target_row["reply_to"]):
        #print(id, target_row["speaker"], target_row["id"], target_row["reply_to"])

        word_set = set(sentence_split)
        transcript_set = set(row_split)

        if transcript_set.issubset(word_set):
            print(id, target_row["speaker"], target_row["id"], target_row["reply_to"])
            print(target_row["clean_text"],"\n", row["transcript"])

            audio_array = row["audio.bytes"]
            #sampling_rate = row['sampling_rate']

            #sf.write("./data_cleaning/swda_audio/" + row['audio.path'], audio_array, sampling_rate)
            with open("./data_cleaning/swda_audio/" + row['audio.path'], mode='bw') as f:
                f.write(audio_array)

            final_audio = final_audio + ", " + row['audio.path']
            #return row['audio.path']

    return final_audio

filtered_utt_df["found"] = filtered_utt_df.apply(lambda x: find_audio(x, partition.iterrows()), axis=1)

filtered_utt_df.to_csv("./data_cleaning/swda_declarative.csv")
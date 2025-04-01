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
        #print(sentence)
        #if "qy^d" in sentence:
        if "sd" in sentence or "sv" in sentence:
            #print(sentence)
            return utt
        
def clean_text(text):
    split_text = text.split()
    #print(split_text)

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

    filtered_utt_df.to_csv("./data_cleaning/swda_statements.csv")

create_csv()
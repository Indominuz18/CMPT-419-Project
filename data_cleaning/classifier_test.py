from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

file_name = "swda_declarative_final.csv"
question_utt_df = pd.read_csv("./data_cleaning/swda_csv_files/" + file_name)

def predict(row):
    text = row["clean_text"] #.replace("?", "")
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()

    prob_array = probs.numpy()[0]
    row["class"] = predicted_class
    row["prob_statement"] = prob_array[0]
    row["prob_question"] = prob_array[1]

    # print(predicted_class)
    return row

# Test off the shelf question vs statement classifiers
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")

question_utt_df2 = question_utt_df.apply(predict, axis=1)
stats = question_utt_df2.groupby(['class']).size()
print("shahrukhx01/question-vs-statement-classifier accuracy: " + str(stats[1] / (stats[0] + stats[1])))


tokenizer = AutoTokenizer.from_pretrained("mrsinghania/asr-question-detection")
model = AutoModelForSequenceClassification.from_pretrained("mrsinghania/asr-question-detection")

question_utt_df2 = question_utt_df.apply(predict, axis=1)
stats = question_utt_df2.groupby(['class']).size()
print("mrsinghania/asr-question-detection accuracy: " + str(stats[1] / (stats[0] + stats[1])))
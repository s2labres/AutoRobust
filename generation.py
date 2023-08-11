import json
import pandas as pd
import random
import string
import csv
import re
from collections import defaultdict
import nltk
nltk.download('words')
from nltk.corpus import words

base_path = "/home/kylesa/avast_clf/v0.2/"
settings = {
    "corpus": base_path + "/corpus.csv",
    "target_vals": base_path + "/sum.json",
    "labels": base_path + "data_avast/labels.csv",
    "report_folder": base_path + "data_avast/",
    "adv_folder": base_path + "data_smp/"}

"""
Modification rules, given an initial clean report from the source class:
1: Add an amount of entries from the target class, individually per key.
2: Modify an amount of entries per key, by replacing words with:
a) English words of fixed length (5-10).
b) Words occuring in the corpus of the target class.
c) Random ASCII string of fixed length (4-8).

Parameters:
mod_ratio -- the percentage of values for each key that get modified
add_ratio -- the max ratio wrt to the total of values added
"""

# Load target class corpus
with open(settings["corpus"], 'r') as file:
    reader = csv.reader(file)
    corpus = list(set([row[0] for row in reader]))
corpus = [string for string in corpus if ':' not in string and len(string)<30]
# Load english corpus
awords = [w for w in words.words() if 5 <= len(w) <= 10]
# Load target class entries
trt = json.load(open(settings["target_vals"]))
# print(trt)
# for key, value in trt:
#     for i in range(0, len(value)):
#         value[i] = value[1].replace('\\\\','\\')
# Define possible path separators
seps = ["\\","/"]

def make_advs(mod_ratio=0.8, add_ratio=1):
    df_labels = pd.read_csv(settings["labels"])
    df = df_labels[df_labels["family"] == "virlock"]
    files = [base_path + "data_avast/" + v + ".json" for v in df.iloc[:, 0]]
    for i, file in enumerate(files):
        with open(file) as f:
            rep = json.load(f)["summary"]
            # Modify
            for key, value in rep.items():
                # print(len(value),'\n')
                if key in ["resolved_apis"]:
                    continue # We are not allowed to modify apis
                ln = int(random.uniform(0, 1) * mod_ratio * len(value))
                for j in range(0, ln):
                    # value[j] = value[j].replace(,'\\')
                    # value[j] = value[j].replace('\\\\','\\')
                    for separator in seps:
                        if separator in value[j]:
                            # print(value[i])
                            value[j] = replace_words(value[j], separator)
            # Add
            for key, value in rep.items():
                mx1 = len(trt[key])
                mx2 = max(len(value), 7)
                ax = random.uniform(0, 1) * add_ratio * mx2
                lng = min(int(ax), mx1)
                if mx1 == 1: lng = random.choice([0,1])
                # print(mx1, lng, ax, len(value))
                # for _ in range(lng):                
                value.extend(random.sample(trt[key], lng))
        # for key, value in rep.items():
        #     print(len(value))
        rep = {"summary": rep}
        with open(settings["adv_folder"] +  df.iloc[i, 0] + ".json", "w") as outfile:
                json.dump(rep, outfile)

def replace_words(s, sep):
    parts = s.split(sep)
    # print(repr(sep), repr(parts), repr(s))                               
    if len(parts) > 2:
        # Replace words from the index
        for i in range(2, len(parts)):
            parts[i] = get_word()
        return sep.join(parts)
    elif len(parts) == 1:
        return get_word()
    else:
        return get_word() + sep + get_word()

def get_word():
    ch = random.choice([1,2,3])
    if ch == 1:
        word = random.choice(corpus)
    elif ch == 2:
        word = random.choice(awords)
    else:
        lgth = random.randrange(4, 10)
        word = ''.join(random.choices(string.ascii_letters + string.digits, k=lgth))
    return word

if __name__ == "__main__":
    make_advs()
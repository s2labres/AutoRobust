import json
import re
import csv
import pandas as pd
from collections import defaultdict, OrderedDict

base_path = "/home/kylesa/avast_clf/v0.2/"
settings = {
    "labels": base_path + "data_avast/labels.csv",
    "report_folder": base_path + "data_avast/",
    "save_file": base_path + "sum.json",
    "corpus": base_path + "corpus.json"
}

keyser = [
    "keys",
    "resolved_apis",
    "executed_commands",
    "write_keys",
    "files",
    "read_files",
    "started_services",
    "created_services",
    "write_files",
    "delete_keys",
    "read_keys",
    "delete_files",
    "mutexes"
]

def get_most_frequent(reports, maxval):
    sumrep = defaultdict(list)
    for report in reports:
        with open(report) as f:
            trep = json.load(f)
        # Iterate over the keys in the current dictionary
        for key, values in trep["summary"].items():
            # If the key doesn't exist in the result dictionary yet, add it with an empty array
            if key not in sumrep:
                sumrep[key] = []
            # Append the values in the current dictionary to the array in the result dictionary
            sumrep[key].extend(values)
            # append!(result_dict[key], values...)

    freqrep = {}
    for keys in keyser:
        freq = defaultdict(int)
        for value in sumrep[keys]:
            freq[value] += 1
        # Get the values of the dictionary as an array
        vl = list(freq.values())
        sorted_values = sorted(vl, reverse=True)
        top_values = sorted_values[:maxval]
        # print(keys, sorted_values)
        top_set = set(top_values)
        # print(top_values, top_set)
        top_keys = []

        # Iterate over the key-value pairs in the dictionary
        for key, value in freq.items():
            # If the current value is in the top values, append the key to the top_keys array
            if value in top_set:
                top_keys.append(key)
        freqrep[keys] = top_keys

    for key, value in freqrep.items():
        print(key, len(value))

    with open(settings["save_file"], "w") as f:
        json.dump(freqrep, f)

# def get_corpus():
#     corpus = set()
#     df_labels = pd.read_csv(settings["labels"])
#     df = df_labels[df_labels["family"] == "virlock"]
#     files = [base_path + "data_avast/" + v + ".json" for v in df.iloc[:, 0]]

#     for file in files:
#         with open(file) as f:
#             rep = json.load(f)
#             for key in rep["summary"]:
#                 for value in rep["summary"][key]:
#                     # split the value by backslash, period, or use the full string if no backslash or period
#                     tokens = [x for x in re.split(r"\\\\|\\|\.|/", value) if x]
#                     # add each token to the corpus
#                     for token in tokens:
#                         corpus.add(token)
#     with open(settings["corpus"], 'w', newline='') as file:
#         writer = csv.writer(file)
#         for item in corpus:
#             writer.writerow([item])


def get_corpus():
    corpus = defaultdict(int)  # Initialize corpus as a defaultdict with int values
    df_labels = pd.read_csv(settings["labels"])
    df = df_labels[df_labels["family"] == "virlock"]
    files = [base_path + "data_avast/" + v + ".json" for v in df.iloc[:, 0]]

    for file in files:
        with open(file) as f:
            rep = json.load(f)
            for key in rep["summary"]:
                for value in rep["summary"][key]:
                    # split the value by backslash, period, or use the full string if no backslash or period
                    tokens = [x for x in re.split(r"\\\\|\\|\.|/", value) if x]
                    # add each token to the corpus and increment its frequency
                    for token in tokens:
                        corpus[token] += 1
    # with open(settings["corpus"], 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for item, freq in corpus.items():
    #         writer.writerow([item, freq])

    rem = ['C:', '*', 'Windows', '1', 'exe', 'dll', 'HKEY_LOCAL_MACHINE']
    # Remove multiple keys from dictionary
    for key in rem:
        del corpus[key]

    # sorted_keys = sorted(corpus, key=corpus.get, reverse=True)
    sorted_dict = dict(sorted(corpus.items(), key=lambda x: x[1], reverse=True))
    count = 0
    for key, value in sorted_dict.items():
        print(f"{key}: {value}")
        count += 1
        if count == 5000:
            break

    # Write corpus to JSON file
    with open(settings["corpus"], 'w') as f:
        json.dump(sorted_dict, f)

def get_sum():
    base = "/home/kylesa/avast_clf/v0.2/data_avast/"
    df_labels = pd.read_csv(settings["labels"])
    df = df_labels[df_labels["family"] == "virlock"]
    files = [base + v + ".json" for v in df.iloc[:, 0]]
    get_most_frequent(files, 300)


if __name__ == "__main__":
    # get_sum()
    get_corpus()
    # with open(settings["corpus"], 'r') as file:
    #     reader = csv.reader(file)
    #     # Create a new set variable and add each row in the CSV file to the set
    #     my_set = set([row[0] for row in reader])
    #     print(my_set)
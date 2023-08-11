import torch
import pandas as pd
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from torch.optim import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base_path = "/home/kylesa/avast_clf/v0.2/"
settings = {
    "labels": base_path + "data_avast/labels.csv",
    "report_folder": base_path + "data_avast/",
    "adv_folder": base_path + "data_smp/"}

# Load the pretrained GPT-2 model
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the tokenizer and data collator
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 1024
max_len = 1024
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def load_reports():
    source_reports = []
    target_reports = []
    df_labels = pd.read_csv(settings["labels"])
    df = df_labels[df_labels["family"] == "virlock"]#.sample(frac=0.05, random_state=42)
    filenames = [v for v in df.iloc[:, 0]]
    files = sorted([settings["report_folder"] + v + ".json" for v in filenames])
    afiles = sorted([settings["adv_folder"] + v + ".json" for v in filenames])
    for file, afile in zip(files, afiles):
        # print(len(target_reports))
        with open(file) as f, open(afile) as g:
            # rep = json.load(f)["summary"]
            # rep = json.dumps(rep)
            # arep = json.load(g)
            # arep = json.dumps(arep)
            crep = f.read()
            arep = g.read()
            crep = crep.replace('\\\\','\\')
            # crep = {'a': 1, 'b': 2}
            arep = arep.replace('\\\\','\\')
            # dataset.append({"clean_report": rep, "adversarial_report": arep})
            source_reports.append(crep)
            target_reports.append(arep)
            # if files[i].split('/')[-1]!= afiles[i].split('/')[-1]: print(files[i])
    return source_reports, target_reports

clean_reports, adv_reports = load_reports()

train_size = int(0.8 * len(clean_reports))
print(train_size)
train_clean, val_clean = clean_reports[:train_size], clean_reports[train_size:]
train_adv, val_adv = adv_reports[:train_size], adv_reports[train_size:]
# print(len(train_clean), len(val_clean), len(train_adv), len(val_adv))
print("Phase 0 done!")
train_clean_enc = tokenizer(train_clean, truncation=True, padding=True)
# train_clean_enc = tokenizer.encode_plus(train_clean, 
#                                         max_length=max_len, 
#                                         truncation=True, 
#                                         padding='max_length')
print("Phase 1 done!")
train_adv_enc = tokenizer(train_adv, truncation=True, padding=True)
# train_adv_enc = tokenizer.encode_plus(train_adv, 
#                                         max_length=max_len, 
#                                         truncation=True, 
#                                         padding='max_length')
print("Phase 2 done!")
val_clean_enc = tokenizer(val_clean, truncation=True, padding=True)
# val_clean_enc = tokenizer.encode_plus(val_clean, 
#                                         max_length=max_len, 
#                                         truncation=True, 
#                                         padding='max_length')
print("Phase 3 done!")
val_adv_enc = tokenizer(val_adv, truncation=True, padding=True)
# val_adv_enc = tokenizer.encode_plus(val_adv, 
#                                         max_length=max_len, 
#                                         truncation=True, 
#                                         padding='max_length')
print("Phase 4 done!")

class ReportDataset(Dataset):
    def __init__(self, clean_reports, adv_reports):
        self.clean_reports = clean_reports
        self.adv_reports = adv_reports

    def __len__(self):
        return len(self.clean_reports)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.clean_reports.items()}
        # print(idx, item)
        # print(self.adv_reports.items()[idx])
        item['labels'] = torch.tensor(self.adv_reports['input_ids'][idx])
        # for val in item.values():
        #     print(len(val))
        return item

train_data = ReportDataset(train_clean_enc, train_adv_enc)
val_data = ReportDataset(val_clean_enc, val_adv_enc)

# Define the hyperparameters for fine tuning
batch_size = 8
learning_rate = 5e-5
num_epochs = 5
warmup_steps = 500
adam_epsilon = 1e-8

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1000,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    gradient_accumulation_steps=1,
    run_name='fine-tuning on adversarial dataset'
)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

# Define the scheduler
num_training_steps = len(train_data) * training_args.num_train_epochs // training_args.per_device_train_batch_size
warmup_steps = int(num_training_steps * 0.1)  # 10% of total training steps
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
    eval_dataset=val_data,
    optimizers=[optimizer, scheduler],
)

trainer.train()

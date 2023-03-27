# -*- coding: utf-8 -*-

from .configuration import CONLLIOBV2, MODEL_NAME, config, files_configs
from reader import correct_file, readfile, dataframe_from_reader
from metric.metric import SpanF1, SpanF1_fix
from models import RemBertForTokenClassification

import os
import wandb
import pandas as pd
import numpy as np
import torch
import pprint
import seqeval

from transformers import RemBertForTokenClassification, RemBertTokenizerFast, RemBertConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch import nn
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report



correct_file(files_configs["train_path"])
correct_file(files_configs["test_path"])
train_reader = readfile(files_configs["train_path"])
test_index, test_reader = readfile(files_configs["test_path"], return_index_file=True)
train_data = dataframe_from_reader(train_reader)
test_data = dataframe_from_reader(test_reader)

labels_to_ids  = CONLLIOBV2
ids_to_labels = {k:v for v, k in labels_to_ids.items()}
set_unique_labels = set(labels_to_ids.keys())


print("base model name - ", MODEL_NAME)

os.environ["WANDB_MODE"]="offline"
wandb.init(
  project="NER multilangual",
  notes= files_configs["wandb_notes"],
  name = files_configs["wandb_run_name"],
  config=config,
)


class BertModelforNer(torch.nn.Module):
    def __init__(self):
        super(BertModelforNer, self).__init__()
        self.bert = RemBertForTokenClassification.from_pretrained(files_configs["base_model_path"], num_labels=len(set_unique_labels))
        self.bert.config.label2id = labels_to_ids
        self.bert.config.id2label = ids_to_labels
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        return output

    
model = BertModelforNer()
tokenizer = RemBertTokenizerFast.from_pretrained(files_configs["base_model_path"])

label_all_tokens = False
def aling_label(texts, labels):
    tokenized_inputs = tokenizer(texts.split(), is_split_into_words=True, padding='max_length', max_length=config["max_length"], truncation=True)
    words_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in words_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100 )
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids

class DataSequence(torch.utils.data.Dataset):
    def __init__(self, dataframe):

        labels = [line.split() for line in dataframe['labels'].values.tolist()]
        txts = dataframe["text"].values.tolist()
        self.texts = [tokenizer(i.split(),is_split_into_words=True, padding='max_length', max_length=config["max_length"], truncation=True, return_tensors="pt") for i in txts]
        
        self.labels = [aling_label(i,j) for i,j in zip(txts, labels)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])
    
    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return (batch_data, batch_labels)


df_train, df_val = np.split(train_data.sample(frac=1, random_state=1),
                                   [int(config["TRAIN_VAL_SPLIT"]* len(train_data))])
df_train

def conv_ids_to_label(ids_tensor):
    return ' '.join([ids_to_labels[int(i)] for i in ids_tensor])

def train_loop(model, df_train, df_val):
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=config["BATCH_SIZE"])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = AdamW(model.parameters(), lr= config["LEARINGIN_RATE"])
    total_steps = len(train_dataloader) * config["EPOCHS"]

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = config["num_warmup_steps"],
                                                num_training_steps = total_steps,
                                                num_cycles=config["num_cycles"])
    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000
    for epoch_num in range(config["EPOCHS"]):
        total_acc_train = 0
        total_loss_train = 0

        model.train()
        metrics = SpanF1()
        for idx, (train_data, train_label) in enumerate(tqdm(train_dataloader)):
            train_label = train_label.to(device)
            mask = train_data["attention_mask"].squeeze(1).to(device)
            input_id = train_data["input_ids"].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)
            for i in range(logits.shape[0]):
            
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()

            loss.backward()
            
            if config["USE_CLIP_GRAD"]:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=config["CLIP_GRAD_VALUE"])
            optimizer.step()
            scheduler.step()
            
            current_steps = config['BATCH_SIZE'] * (idx + 1)
            if (idx+1) % 500 == 0:
                print('train_total_epochs_loss = ', total_loss_train / current_steps)
                log_dict =  {
                       "train_curent_batch_acc": acc,
                       "train_loss": loss,
                       "epoch": epoch_num,
                       "learnig_rate": (scheduler.get_lr()[0]),
                       "train_total_epoch_acc": total_acc_train / current_steps,
                       "train_total_epoch_loss": total_loss_train / current_steps,
                       "train_example": wandb.Html("""
                                               True labels: <p>{0} </p>
                                               Pred labels: <p>{1}</p>
                                               """.format(conv_ids_to_label(label_clean),conv_ids_to_label(predictions))
                                            )     
                }
            else:
                log_dict = {
                       "train_curent_batch_acc": acc,
                       "train_loss": loss,
                       "epoch": epoch_num,
                       "learnig_rate": (scheduler.get_lr()[0]),
                       "train_total_epoch_acc": total_acc_train / current_steps,
                       "train_total_epoch_loss": total_loss_train / current_steps
                }
            wandb.log(log_dict)

        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():
            all_true_labels = []
            all_predictions = []
            for idx, (val_data, val_label) in enumerate(val_dataloader):

                val_label = val_label.to(device)
                mask = val_data["attention_mask"].squeeze(1).to(device)
                input_id = val_data["input_ids"].squeeze(1).to(device)

                loss, logits = model(input_id, mask, val_label)

                for i in range(logits.shape[0]):
                    logits_clean = logits[i][val_label[i] != -100]
                    label_clean = val_label[i][val_label[i] != -100]
                    predictions = logits_clean.argmax(dim=1)
                    
                    if idx == 0 and i == 0:
                        print("label: ", label_clean)
                        print("pred: ", predictions)
                    prediction_label = [ids_to_labels[int(i)] for i in predictions]
                    label_label = [ids_to_labels[int(i)] for i in label_clean]
                    
                    all_true_labels.append(label_label)
                    all_predictions.append(prediction_label)
                    metrics([prediction_label],[label_label])
                    
                    acc = (predictions == label_clean).float().mean()
                    total_acc_val += acc
                    total_loss_val += loss.item()
                current_steps = config['BATCH_SIZE'] * (idx + 1)
                if (idx+1) % 50 == 0:
                    log_dict = {
                           "valid_curent_batch_acc": acc,
                           "valid_loss": loss,
                           "valid_total_epoch_acc": total_acc_val / current_steps,
                           "valid_total_epoch_loss": total_loss_val / current_steps,
                           "valid_example": wandb.Html("""
                                                   True labels: <p>{0} </p>
                                                   Pred labels: <p>{1}</p>
                                                   """.format(conv_ids_to_label(label_clean),conv_ids_to_label(predictions)))
                    }
                else:
                    log_dict = {
                           "valid_curent_batch_acc": acc,
                           "valid_loss": loss,
                           "valid_total_epoch_acc": total_acc_val / current_steps,
                           "valid_total_epoch_loss": total_loss_val / current_steps
                    }
                wandb.log(log_dict)
        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        wandb.log({
            "f1_score": f1_score(all_true_labels, all_predictions),
            "recall_score": recall_score(all_true_labels, all_predictions),
            "precision_score": precision_score(all_true_labels, all_predictions),
            "accuracy_score": accuracy_score(all_true_labels, all_predictions)
            
        })
        
        pprint.pprint(metrics.get_metric(True))
        print("\n"*2)
        print("\n"*2)
        print(classification_report(all_true_labels, all_predictions))
        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}'
        )
        model_name_check_point = f"./model_epoch_num_{epoch_num + 1}"
        model.bert.save_pretrained(model_name_check_point)
        tokenizer.save_pretrained(model_name_check_point)


res = train_loop(model, df_train, df_val)

wandb.finish()

model.bert.save_pretrained(files_configs["res_path"])
tokenizer.save_pretrained(files_configs["res_path"])


def evaluate(model, df_test_data):
    metrics = SpanF1()
    test_dataset = DataSequence(df_test_data)

    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=config["BATCH_SIZE"])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0
    with torch.no_grad():
        for test_data, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)

            input_id = test_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):

                logits_clean = logits[i][test_label[i] != -100]
                label_clean = test_label[i][test_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)

                prediction_label = [ids_to_labels[int(i)] for i in predictions]
                label_label = [ids_to_labels[int(i)] for i in label_clean]
                metrics([prediction_label],[label_label])

                acc = (predictions == label_clean).float().mean()
                total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test_data)
    print(f'Test Accuracy: {total_acc_test / len(df_test_data): .3f}')
    return metrics.get_metric(True)

metric_table = evaluate(model, test_data)

metric_table

def align_word_ids(texts):
  
    
    tokenized_inputs = tokenizer(texts, is_split_into_words=True, padding='max_length', max_length=256, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    input = sentence.split(' ')
    text = tokenizer(input, is_split_into_words=True, padding='max_length', max_length = 256, truncation=True, return_tensors="pt")

    mask = text['attention_mask'][0].unsqueeze(0).to(device)

    input_id = text['input_ids'][0].unsqueeze(0).to(device)
    label_ids = torch.Tensor(align_word_ids(input)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]


    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    # print(sentence)
    # print(prediction_label)
    return prediction_label
            
evaluate_one_text(model, "Elon Musk 's brother sits on the boards of Tesla")

LINE = 0
print(test_data["text"].values.tolist()[LINE])
print(test_data["labels"].values.tolist()[LINE].split())
print("----")
print(evaluate_one_text(model, test_data["text"].values.tolist()[LINE]))

test_data

results = []
for i in range(test_data.shape[0]):
    text = test_data.iloc[i]['text'].translate({769: '_', 1620: '_', 1611: '_', 65533: '_', 1616: '_', 3657: '_', 3633: '_', 3637: '_', 2492: '_', 8203: '_', 8204: '_', 8205: '_', 8206: '_'})
    results.append(evaluate_one_text(model, text))

with open("./multi.pred.conll", "w") as my_file:
    my_file.write("\n")
    for i in range(len(test_index)):
        my_file.write(test_index[i])
        for j in results[i]:
            my_file.write("      "+ j +"\n")
        my_file.write("\n")
        my_file.write("\n")


"""# NEw test"""

answers = test_data.labels.to_list()

metrics = SpanF1_fix()
for it1, it2 in zip(results, answers):
    pred = it1
    label = it2.split()
    metrics([pred],[label])
metrics.get_metric()


answers = [line.split() for line in answers]

for i in range(len(results)):
    if len(answers[i]) != len(results[i]):
        print(f'-{i}-')

print(classification_report(answers, results, digits=5))

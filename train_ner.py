# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import wandb

CONLLIOBV2 = {
    "B-Disease":0,
    "I-Disease":1,
    "B-Symptom":2,
    "I-Symptom":3,
    "B-AnatomicalStructure":4,
    "I-AnatomicalStructure":5,
    "B-MedicalProcedure":6,
    "I-MedicalProcedure":7,
    "B-Medication/Vaccine":8,
    "I-Medication/Vaccine":9,

    "B-OtherPROD":10,
    "I-OtherPROD":11,
    "B-Drink":12,
    "I-Drink":13,
    "B-Food":14,
    "I-Food":15,
    "B-Vehicle":16,
    "I-Vehicle":17,
    "B-Clothing":18,
    "I-Clothing":19,

    "B-OtherPER":20,
    "I-OtherPER":21,
    "B-SportsManager":22,
    "I-SportsManager":23,
    "B-Cleric":24,
    "I-Cleric":25,
    "B-Politician":26,
    "I-Politician":27,
    "B-Athlete":28,
    "I-Athlete":29,
    "B-Artist":30,
    "I-Artist":31,
    "B-Scientist":32,
    "I-Scientist":33,
    
    "B-ORG":34,
    "I-ORG":35,
    "B-TechCorp":36,
    "I-TechCorp":37,
    "B-CarManufacturer":38,
    "I-CarManufacturer":39,
    "B-SportsGRP":40,
    "I-SportsGRP":41,
    "B-AerospaceManufacturer":42,
    "I-AerospaceManufacturer":43,
    "B-OtherCorp":44,
    "I-OtherCorp":45,
    "B-PrivateCorp":46,
    "I-PrivateCorp":47,
    "B-PublicCorp":48,
    "I-PublicCorp":49,
    "B-MusicalGRP":50,
    "I-MusicalGRP":51,

    "B-OtherCW":52,
    "I-OtherCW":53,
    "B-Software":54,
    "I-Software":55,
    "B-ArtWork":56,
    "I-ArtWork":57,
    "B-WrittenWork":58,
    "I-WrittenWork":59,
    "B-MusicalWork":60,
    "I-MusicalWork":61,
    "B-VisualWork":62,
    "I-VisualWork":63,

    "B-Station":64,
    "I-Station":65,
    "B-HumanSettlement":66,
    "I-HumanSettlement":67,
    "B-OtherLOC":68,
    "I-OtherLOC":69,
    "B-Facility":70,
    "I-Facility":71,

    'O': 72
}

import pandas as pd
import numpy as np
import torch
import pprint
import seqeval
from transformers import RemBertForTokenClassification, RemBertTokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch import nn
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

# !cp /kaggle/input/dataset-for-ner-competition/multi_train.conll .
# !cp /kaggle/input/dataset-for-ner-competition/multi_dev.conll .

def correct_file(path: str):
    flag = True
    with open(path, 'r') as f:
        if f.read()[-3:] == "\n\n\n":
            flag = False 
    if flag:
        with open(path, 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write('\n')
        print(f"file corrected")
def readfile(filename, return_index_file=False):
    '''
    read data from file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    sentence_ids = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence[1:], label[1:]))
                sentence_ids.append(sentence[0] + " id " + label[0] +"\n")
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        # print(splits)

        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    if return_index_file:
        return sentence_ids, data
    return data

correct_file("./public_data/MULTI_Multilingual/multi_train.conll")
correct_file("./public_data/MULTI_Multilingual/multi_dev.conll")
train_reader = readfile("./public_data/MULTI_Multilingual/multi_train.conll")
test_index, test_reader = readfile("./public_data/MULTI_Multilingual/multi_dev.conll", return_index_file=True)

#!git lfs install
#!git clone https://huggingface.co/google/rembert

def dataframe_from_reader(reader):
    sentences_one_line = []
    sentences = []
    labels_one_line = []
    labels = []
    for line in reader:
        sentence = " ".join(line[0])
        label = " ".join(line[1])
        sentences_one_line.append(sentence)
        sentences.append(line[0])
        labels_one_line.append(label)
        labels.append(line[1])
    data = pd.DataFrame({"text": sentences_one_line, "labels": labels_one_line})
    return data
train_data = dataframe_from_reader(train_reader)
test_data = dataframe_from_reader(test_reader)

labels_to_ids  = CONLLIOBV2
ids_to_labels = {k:v for v, k in labels_to_ids.items()}
set_unique_labels = set(labels_to_ids.keys())

MODEL_NAME = 'google/rembert'

config = dict(
    model_name = MODEL_NAME,
    LEARINGIN_RATE = 1e-5,
    EPOCHS = 3,
    BATCH_SIZE = 14,
    TRAIN_VAL_SPLIT = 0.8,
    num_warmup_steps = 3000,
    CLIP_GRAD_VALUE = 5,
    USE_CLIP_GRAD = True,
    dict_for_labels = CONLLIOBV2,
    optimizer = "AdamW",
    max_length = 128
)

print(MODEL_NAME)

import os
os.environ["WANDB_MODE"]="offline"

wandb.init(
  project="NER multilangual",
  notes="rembert ",
  name = "google/rembert(test_resuorse)",
  config=config,
)

from transformers import RemBertConfig
class BertModel(torch.nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        config = RemBertConfig.from_json_file("./rembert/config.json")
        self.bert = RemBertForTokenClassification.from_pretrained("./rembert",
                                                               num_labels=len(set_unique_labels))
        self.bert.config.label2id = labels_to_ids
        self.bert.config.id2label = ids_to_labels
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask,
                           labels=label, return_dict=False)
        return output
model = BertModel()

tokenizer = RemBertTokenizerFast.from_pretrained("./rembert")

label_all_tokens = False

def aling_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding="max_length", max_length=config["max_length"], truncation=True)
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
        txt = dataframe["text"].values.tolist()
        self.texts = [tokenizer(str(i), padding="max_length", max_length=config["max_length"], truncation=True, return_tensors="pt") for i in txt]
        
        self.labels = [aling_label(i,j) for i,j in zip(txt, labels)]

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

class SpanF1():
    def __init__(self, non_entity_labels=['O']) -> None:
        """
        class for calculating NER score for all tokens
        """
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP, self._FP, self._GT = defaultdict(int), defaultdict(int), defaultdict(int)
        self.non_entity_labels = set(non_entity_labels)

    def __call__(self, batched_predicted_spans, batched_gold_spans, sentences=None):
        non_entity_labels = self.non_entity_labels

        for predicted_spans, gold_spans in zip(batched_predicted_spans, batched_gold_spans):
            gold_spans_set = set([x for x in gold_spans])
            pred_spans_set = set([x for x in predicted_spans])

            self._num_gold_mentions += len(gold_spans_set)
            self._num_recalled_mentions += len(gold_spans_set & pred_spans_set)
            self._num_predicted_mentions += len(pred_spans_set)

            for val in gold_spans:
                if val not in non_entity_labels:
                    self._GT[val] += 1

            for idx, val in enumerate(predicted_spans):
                if val in non_entity_labels:
                    continue
                if val in gold_spans and val == gold_spans[idx]:
                    self._TP[val] += 1
                else:
                    self._FP[val] += 1

    def get_metric(self, reset: bool = False) -> float:
        all_tags = set()
        all_tags.update(self._TP.keys())
        all_tags.update(self._FP.keys())
        all_tags.update(self._GT.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self.compute_prf_metrics(true_positives=self._TP[tag],
                                                                     false_negatives=self._GT[tag] - self._TP[tag],
                                                                     false_positives=self._FP[tag])
            all_metrics['P@{}'.format(tag)] = precision
            all_metrics['R@{}'.format(tag)] = recall
            all_metrics['F1@{}'.format(tag)] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self.compute_prf_metrics(true_positives=sum(self._TP.values()),
                                                                 false_positives=sum(self._FP.values()),
                                                                 false_negatives=sum(self._GT.values())-sum(self._TP.values()))
        all_metrics["micro@P"] = precision
        all_metrics["micro@R"] = recall
        all_metrics["micro@F1"] = f1_measure

        if self._num_gold_mentions == 0:
            entity_recall = 0.0
        else:
            entity_recall = self._num_recalled_mentions / float(self._num_gold_mentions)

        if self._num_predicted_mentions == 0:
            entity_precision = 0.0
        else:
            entity_precision = self._num_recalled_mentions / float(self._num_predicted_mentions)

        all_metrics['MD@R'] = entity_recall
        all_metrics['MD@P'] = entity_precision
        all_metrics['MD@F1'] = 2. * ((entity_precision * entity_recall) / (entity_precision + entity_recall + 1e-13))
        all_metrics['ALLTRUE'] = self._num_gold_mentions
        all_metrics['ALLRECALLED'] = self._num_recalled_mentions
        all_metrics['ALLPRED'] = self._num_predicted_mentions
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def compute_prf_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP.clear()
        self._FP.clear()
        self._GT.clear()

df_train, df_val = np.split(train_data.sample(frac=1, random_state=84),
                                   [int(config["TRAIN_VAL_SPLIT"]* len(train_data))])
df_train

def conv_ids_to_label(ids_tensor):
    return ' '.join([ids_to_labels[int(i)] for i in ids_tensor])

def train_loop(model, df_train, df_val):
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=config["BATCH_SIZE"],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=config["BATCH_SIZE"])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = AdamW(model.parameters(), lr= config["LEARINGIN_RATE"])
    total_steps = len(train_dataloader) * config["EPOCHS"]

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = config["num_warmup_steps"],
                                                num_training_steps = total_steps)
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
res = train_loop(model, df_train, df_val)

wandb.finish()

# xlm-roberta-base


res_path = "google/rembert-ft_for_multi_ner_v2"
model.bert.save_pretrained(res_path)
tokenizer.save_pretrained(res_path)
!zip -r "google/rembert-ft_for_multi_ner.zip" google//rembert-ft_for_multi_ner_v2/

def evaluate(model, df_test_data):
    metrics = SpanF1()
    test_dataset = DataSequence(df_test_data)

    test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=12)

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
    text = test_data.iloc[i]['text'].replace('\u200b','x')
    results.append(evaluate_one_text(model, text))

with open("./multi.pred.conll", "w") as my_file:
    my_file.write("\n")
    for i in range(len(test_index)):
        my_file.write(test_index[i])
        for j in results[i]:
            my_file.write("      "+ j +"\n")
        my_file.write("\n")
        my_file.write("\n")

!zip my_submission.zip "./multi.pred.conll"

"""# NEw test"""

class SpanF1_fix():
    def __init__(self, non_entity_labels=['O']) -> None:
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP, self._FP, self._GT = defaultdict(int), defaultdict(int), defaultdict(int)
        self.non_entity_labels = set(non_entity_labels)

    def __call__(self, batched_predicted_spans, batched_gold_spans, sentences=None):
        non_entity_labels = self.non_entity_labels

        for predicted_spans, gold_spans in zip(batched_predicted_spans, batched_gold_spans):
            gold_spans_set = set([x if x == "O" else x[2:] for x in gold_spans])
            pred_spans_set = set([x if x == "O" else x[2:] for x in predicted_spans])
            self._num_gold_mentions += len(gold_spans_set)
            self._num_recalled_mentions += len(gold_spans_set & pred_spans_set)
            self._num_predicted_mentions += len(pred_spans_set)

            for val in gold_spans:
                if val not in non_entity_labels:
                    self._GT[val[2:]] += 1

            for idx, val in enumerate(predicted_spans):
                # print(idx, "----", val)
                # print(val in gold_spans)
                # print(val == gold_spans[idx])
                if val in non_entity_labels:
                    continue
                if val in gold_spans and val[2:] == gold_spans[idx][2:]:
                    self._TP[val[2:]] += 1
                else:
                    self._FP[val[2:]] += 1

    def get_metric(self, reset: bool = False) -> float:
        all_tags = set()
        all_tags.update(self._TP.keys())
        all_tags.update(self._FP.keys())
        all_tags.update(self._GT.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self.compute_prf_metrics(true_positives=self._TP[tag],
                                                                     false_negatives=self._GT[tag] - self._TP[tag],
                                                                     false_positives=self._FP[tag])
            all_metrics['P@{}'.format(tag)] = precision
            all_metrics['R@{}'.format(tag)] = recall
            all_metrics['F1@{}'.format(tag)] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self.compute_prf_metrics(true_positives=sum(self._TP.values()),
                                                                 false_positives=sum(self._FP.values()),
                                                                 false_negatives=sum(self._GT.values()) - sum(
                                                                     self._TP.values()))
        all_metrics["micro@P"] = precision
        all_metrics["micro@R"] = recall
        all_metrics["micro@F1"] = f1_measure

        if self._num_gold_mentions == 0:
            entity_recall = 0.0
        else:
            entity_recall = self._num_recalled_mentions / float(self._num_gold_mentions)

        if self._num_predicted_mentions == 0:
            entity_precision = 0.0
        else:
            entity_precision = self._num_recalled_mentions / float(self._num_predicted_mentions)

        all_metrics['MD@R'] = entity_recall
        all_metrics['MD@P'] = entity_precision
        all_metrics['MD@F1'] = 2. * ((entity_precision * entity_recall) / (entity_precision + entity_recall + 1e-13))
        all_metrics['ALLTRUE'] = self._num_gold_mentions
        all_metrics['ALLRECALLED'] = self._num_recalled_mentions
        all_metrics['ALLPRED'] = self._num_predicted_mentions
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def compute_prf_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP.clear()
        self._FP.clear()
        self._GT.clear()

answers = test_data.labels.to_list()

metrics = SpanF1_fix()
for it1, it2 in zip(results, answers):
    pred = it1
    label = it2.split()
    metrics([pred],[label])
metrics.get_metric()

import seqeval

y_pred = [['B-OtherPER', 'I-OtherPER', 'O', 'O', 'O', 'O', 'O', 'B-PublicCorp', 'I-PublicCorp', 'I-ORG',
          'I-PublicCorp']]
y_true =  [['B-OtherPER', 'I-OtherPER', 'O', 'O', 'O', 'O', 'O', 'B-PublicCorp', 'I-PublicCorp', 'I-PublicCorp',
          'I-PublicCorp']]

f1_score(y_true, y_pred)

answers = [line.split() for line in answers]

for i in range(len(results)):
    if len(answers[i]) != len(results[i]):
        print(f'-{i}-')

print(classification_report(answers, results, digits=5))


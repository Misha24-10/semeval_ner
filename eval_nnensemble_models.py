from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import RemBertForTokenClassification, RemBertTokenizerFast
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
import torch
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
import seqeval
from reader import correct_file, readfile, dataframe_from_reader
from datetime import datetime
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from models import Ensemble_model
import torch.optim.lr_scheduler.StepLR

rembert_path = "./fine-tuned-models/google-rembert-ft_for_multi_ner_v3"
xlm_roberta_path = "./fine-tuned-models/xlm_roberta_large_mountain"
rembert_path_2 = "./fine-tuned-models/google-rembert-ft_for_multi_ner_sky"


train_file_path = "./public_data/MULTI_Multilingual/multi_train.conll"
val_file_path = "./public_data/MULTI_Multilingual/multi_dev.conll"


model_1 = RemBertForTokenClassification.from_pretrained(rembert_path)
tokenizer_1 = RemBertTokenizerFast.from_pretrained(rembert_path)

model_2 = XLMRobertaForTokenClassification.from_pretrained(xlm_roberta_path)
tokenizer_2 = XLMRobertaTokenizerFast.from_pretrained(xlm_roberta_path)

model_3 = RemBertForTokenClassification.from_pretrained(rembert_path_2)
tokenizer_3 = RemBertTokenizerFast.from_pretrained(rembert_path_2)

weights = {'model_1': 0.5, 'model_2': 0.5, 'model_3': 0.5}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_1.to(device)
model_2.to(device)
model_3.to(device)

model_1.eval()
model_2.eval()
model_3.eval()


correct_file(train_file_path)
correct_file(val_file_path)
train_index, train_reader = readfile(train_file_path, return_index_file=True)
test_index, test_reader = readfile(val_file_path, return_index_file=True)
train_subbmit = dataframe_from_reader(train_reader)
val_subbmit = dataframe_from_reader(test_reader)


def compute_last_leyer_probs(model, tokenizer, sentence):

    number_of_tokens = tokenizer.encode_plus(sentence, return_tensors='pt',)['input_ids'].shape[-1]
    list_of_words = sentence.split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(list_of_words, is_split_into_words=True, padding='max_length', max_length = min(number_of_tokens,512), truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    label_ids = torch.Tensor(align_word_ids(inputs.word_ids()))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu()
        return (logits[:, (label_ids == 1), :])


def get_logits_concat(sentence):
    try:
        logits1 = compute_last_leyer_probs(model_1, tokenizer_1, sentence)
        logits2 = compute_last_leyer_probs(model_2, tokenizer_2, sentence)
        logits3 = compute_last_leyer_probs(model_3, tokenizer_3, sentence)
        logits = torch.cat((logits1, logits2, logits3), dim=2)
    except:
        print(sentence)
        print(logits1.shape)
        print(logits2.shape)
        print(logits3.shape)
        print(logits = torch.cat((logits1, logits2, logits3), dim=2))
    return logits.reshape(-1, logits.shape[-1])



class DataSequence(torch.utils.data.Dataset):
    def __init__(self, dataframe):

        self.labels = [line.split() for line in dataframe['labels'].values.tolist()]
        self.texts = dataframe["text"].values.tolist()

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return get_logits_concat(self.texts[idx])

    def get_batch_labels(self, idx):
        return torch.Tensor([model_1.config.label2id[i] for i in (self.labels[idx])]).long()
    
    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return {"inputs":batch_data, "labels":batch_labels}


train_dataset = DataSequence(train_subbmit)
val_dataset = DataSequence(val_subbmit)


def custom_collate(data):
    inputs = []
    labels = [] 
    for i in data:
        inputs.append(i["inputs"])
        labels.append(i["labels"])
    inputs = pad_sequence(inputs, batch_first=True)

    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"inputs":inputs, "labels":labels}


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=custom_collate)
val_dataloader = DataLoader(val_dataset, batch_size=32, pin_memory=True, collate_fn=custom_collate)


model_ens = Ensemble_model(num_models=3)
model_ens.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ens.to(device)
optimizer = torch.optim.Adam(model_ens.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
EPOCHS = 1

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_predictions = 0

    for index, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader))):
        inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        loss, logits = model_ens(inputs, labels)
        loss.backward()
        optimizer.step()
        if index % 150 == 0:
            print(f"iteration {index} || current loss = {round(float(loss),3)}")
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    scheduler.step()

    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model_ens.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"model_trained_model_ens{epoch+1}.cpt")


    print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}")



model_ens.eval()
labels_list = []
labels_list_2 = []
for batch in tqdm(val_dataloader, total=len(val_dataloader)):
    inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
    loss, logits = model_ens(inputs, labels)
    labels_clear = []
    for line in labels:
        labels_clear.append([int(i)  for i in line if i != -100])
    res = []
    for i in range(logits.shape[0]):
        res.append(logits[i][:len(labels_clear[i]),:])
    pred_labels = [i.argmax(axis=1) for i in res]

    for line in pred_labels:
        labels_list.append([model_1.config.id2label[int(i)] for i in line])

    for line in labels:
        labels_list_2.append([model_1.config.id2label[int(i)] for i in line if int(i) != -100])


index = 12
print("Pred:", labels_list[index])
print("True:", labels_list_2[index])


print(classification_report(labels_list_2, labels_list, digits=5))


print("Model's state_dict:")
for param_tensor in model_ens.state_dict():
    print(param_tensor, "\t", model_ens.state_dict()[param_tensor].size())



with open("./submission/multi_valid.pred.conll", "w") as my_file:
    my_file.write("\n")
    for i in range(len(labels_list)):
        my_file.write(test_index[i])
        for j in labels_list[i]:
            my_file.write("      "+ j +"\n")
        my_file.write("\n")
        my_file.write("\n")

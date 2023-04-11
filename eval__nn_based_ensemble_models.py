from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import RemBertForTokenClassification, RemBertTokenizerFast
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
import torch
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from reader import correct_file, readfile, dataframe_from_reader
from datetime import datetime
from tqdm.notebook import tqdm
from models import Ensemble_model


rembert_path = "./fine-tuned-models/google-rembert-ft_for_multi_ner_v3"
xlm_roberta_path = "./fine-tuned-models/xlm_roberta_large_mountain"
rembert_path_2 = "./fine-tuned-models/google-rembert-ft_for_multi_ner_sky"

test_file_path = "./public_data/MULTI_Multilingual/multi_dev.conll"
checkpoint_path = "./ensemble_models/model_trained_model_ens_end_epoch_2.cpt"

model_1 = RemBertForTokenClassification.from_pretrained(rembert_path)
tokenizer_1 = RemBertTokenizerFast.from_pretrained(rembert_path)

model_2 = XLMRobertaForTokenClassification.from_pretrained(xlm_roberta_path)
tokenizer_2 = XLMRobertaTokenizerFast.from_pretrained(xlm_roberta_path)

model_3 = RemBertForTokenClassification.from_pretrained(rembert_path_2)
tokenizer_3 = RemBertTokenizerFast.from_pretrained(rembert_path_2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_1.to(device)
model_2.to(device)
model_3.to(device)

model_1.eval()
model_2.eval()
model_3.eval()

correct_file(test_file_path)
test_index, test_reader  = readfile(test_file_path, return_index_file=True)


test_subbmit = dataframe_from_reader(test_reader)

print(test_subbmit)

label_all_tokens = False

def align_word_ids(word_ids, return_word_ids=False):    
    previous_word_idx = None
    label_ids = []
    index_list = []
    for idx, word_idx in enumerate(word_ids):

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
                index_list.append(idx)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    if return_word_ids:
        return label_ids, index_list
    else:
        return label_ids


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


model_ens = Ensemble_model(512)
checkpoint = torch.load(checkpoint_path, map_location=device)
model_ens.load_state_dict(checkpoint['model_state_dict'])


def nn_ensemble_voting(sentence):
    models_logit = get_logits_concat(sentence)
    logits = model_ens(models_logit)
    pred_labels = logits.argmax(axis=1)
    return [model_1.config.id2label[int(i)] for i in pred_labels]


answers = []
size_of_dataset = len(test_subbmit)
for i, line in enumerate(tqdm(test_subbmit.text[:], total=len(test_subbmit))):
    if i % 1_000 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"current step is {i}/{size_of_dataset}, current time: {current_time}")

    answer = nn_ensemble_voting(line.lower())
    answers.append(answer)


with open("./submission/multi.pred.conll", "w") as my_file:
    my_file.write("\n")
    for i in range(len(answers)):
        my_file.write(test_index[i])
        for j in answers[i]:
            my_file.write("      "+ j +"\n")
        my_file.write("\n")
        my_file.write("\n")
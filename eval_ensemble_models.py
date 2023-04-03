from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import RemBertForTokenClassification, RemBertTokenizerFast
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
import torch
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from reader import correct_file, readfile, dataframe_from_reader
from datetime import datetime
from tqdm.notebook import tqdm


rembert_path = "./google-rembert-ft_for_multi_ner_v3"
xlm_roberta_path = "./xlm_roberta_large_for_multi_ner_v3"
test_file_path = "./public_data/MULTI_Multilingual/multi_test.conll"


model_1 = RemBertForTokenClassification.from_pretrained(rembert_path)
tokenizer_1 = RemBertTokenizerFast.from_pretrained(rembert_path)

model_2 = XLMRobertaForTokenClassification.from_pretrained(xlm_roberta_path)
tokenizer_2 = XLMRobertaTokenizerFast.from_pretrained(xlm_roberta_path)
weights = {'model_1': 0.5, 'model_2': 0.5}

model_1.cuda()
model_2.cuda()
model_1.eval()
model_2.eval()


correct_file(test_file_path)
test_index, test_reader  = readfile(test_file_path, return_index_file=True)


val_subbmit = dataframe_from_reader(test_reader)

print(val_subbmit)

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

def evaluate_one_text(tokenizer, model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    input = sentence.split(' ')
    text = tokenizer(input, is_split_into_words=True, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
    label_ids = torch.Tensor(align_word_ids(text.word_ids())).unsqueeze(0).to(device)

    mask = text['attention_mask'][0].unsqueeze(0).to(device)

    input_id = text['input_ids'][0].unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]


    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [model.config.id2label[i] for i in predictions]
    return prediction_label
sent_ex = "Elon Musk 's brother sits on the boards of tesla".lower()
print(sent_ex)
print(evaluate_one_text(tokenizer_1, model_1, sent_ex))

def compute_last_leyer_probs(model, tokenizer, sentence):

    number_of_tokens = tokenizer.encode_plus(sentence, return_tensors='pt',)['input_ids'].shape[-1]
    list_of_words = sentence.split()

    inputs = tokenizer(list_of_words, is_split_into_words=True, padding='max_length', max_length = min(number_of_tokens,512), truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].cuda()
    label_ids = torch.Tensor(align_word_ids(inputs.word_ids()))
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return (logits[:, (label_ids == 1), :])


def weighted_voting(sentence):
    predictions = []
    for idx, (model, tokenizer) in enumerate([(model_1, tokenizer_1), (model_2, tokenizer_2)]):
        logits = compute_last_leyer_probs(model, tokenizer, sentence)
        predictions.append(logits * weights[f'model_{idx+1}'])
    final_logits = sum(predictions)
    final_predictions = torch.argmax(final_logits, dim=2)
    labels = [model_1.config.id2label[i] for i in final_predictions.tolist()[0]]
    return labels


def majority_voting(sentence):
    predictions = []
    for idx, (model, tokenizer) in enumerate([(model_1, tokenizer_1), (model_2, tokenizer_2)]):
        logits = compute_last_leyer_probs(model, tokenizer, sentence)
        labels = torch.argmax(logits, dim=2)
        predictions.append(labels[0].tolist())

    grouped_predictions = list(zip(*predictions))
    final_labels = []
    for group in grouped_predictions:
        final_labels.append(max(set(group), key=group.count))
    final_labels = [model_1.config.id2label[i] for i in final_labels]
    return final_labels

print(majority_voting(sent_ex))



answers = []
size_of_dataset = len(val_subbmit)
for i, line in enumerate(tqdm(val_subbmit.text[:], total=len(val_subbmit))):
    if i % 500 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"current step is {i}/{size_of_dataset}, current time: {current_time}")

    answer = weighted_voting(line.strip().replace('\u200d', 'x').replace('\u200c', 'x').replace('\u200b', 'x').replace('\u200e', 'x'))
    answers.append(answer)


with open("./submission/multi.pred.conll", "w") as my_file:
    my_file.write("\n")
    for i in range(len(answers)):
        my_file.write(test_index[i])
        for j in answers[i]:
            my_file.write("      "+ j +"\n")
        my_file.write("\n")
        my_file.write("\n")
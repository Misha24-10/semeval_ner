import pandas as pd

def correct_file(path: str):
    """
    
    adding empty lines to the end of file 

    """
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
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    if return_index_file:
        return sentence_ids, data
    return data

def dataframe_from_reader(reader):
    sentences_one_line = []
    sentences = []
    labels_one_line = []
    labels = []
    for line in reader:
        sentence = " ".join([s.translate({769: '_', 1620: '_', 1611: '_', 65533: '_', 1616: '_', 3657: '_',
                        3633: '_', 3637: '_', 2492: '_', 8203: '_', 8204: '_', 8205: '_', 8206: '_'}) for s in line[0]])
        label = " ".join(line[1])
        sentences_one_line.append(sentence)
        sentences.append(line[0])
        labels_one_line.append(label)
        labels.append(line[1])
    data = pd.DataFrame({"text": sentences_one_line, "labels": labels_one_line})
    return data
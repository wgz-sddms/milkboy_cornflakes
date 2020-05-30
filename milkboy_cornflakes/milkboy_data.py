## milkboy_data.py

import MeCab
import numpy as np
import pandas as pd
import torch


def nouns_extract(line):
    # MeCab tokenize
    tagger = MeCab.Tagger("")
    keyword=[]
    node = tagger.parseToNode(line).next
    while node:
        if node.feature.split(",")[0] in ["名詞", "動詞", "形容詞"]:
            keyword.append(node.surface)
        node = node.next
    keyword = str(keyword).replace("', '"," ")
    keyword = keyword.replace("\'","")
    keyword = keyword.replace("[","")
    keyword = keyword.replace("]","")
    return keyword

def vectorized_word(words_list, vector_model):
    # word => vector (using vectorize_model)
    input_vector = np.zeros(200)    #dimension is 200 (w2v)
    word_count = 0
    for word in words_list:
        try:
            input_vector += vector_model["[{}]".format(word)]
            word_count += 1
        except KeyError:
            pass
    return input_vector / word_count


class MilkBoyDataSet:
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    @staticmethod
    def from_csv(csv_file):
        df = pd.read_csv(csv_file)
        df = df.iloc[:, 1:3]
        df["text_wakati"] = list(map(lambda text : nouns_extract(text) , df.text))
        labels_list = [label for label in df.label]
        tokens_list = [tokens.split(" ") for tokens in df.text_wakati]
        return MilkBoyDataSet(tokens_list, labels_list)

    def to_model_input(self, w2v):
        x = np.zeros(200)
        for text_wakati in self.tokens:
            X = vectorized_word(text_wakati, w2v)
            x = np.vstack([x,X])
        data_x = x[1:]

        return torch.utils.data.dataset.TensorDataset(
            torch.from_numpy(data_x),
            torch.tensor(self.labels),
        )

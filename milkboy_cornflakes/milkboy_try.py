## milkboy_try.py

import numpy as np
import pickle
import torch
from gensim.models import KeyedVectors
from milkboy_data import MilkBoyDataSet, nouns_extract, vectorized_word
from milkboy_train import NNmodel


def main():
    print("(setting up...)")
    # loading word2vec
    model_dir = './entity_vector/entity_vector.model.bin'
    vector_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)

    # load NNmodel
    net = NNmodel()
    with open("milkboy_model.pickle", "rb") as f:
        net = pickle.load(f)

    i = 0
    while True:
        # get some words
        input_text = input("オカンが言うにはな、 ")
        words_input = nouns_extract(input_text)
        words_input_list = words_input.split(" ")
        vector_input = vectorized_word(words_input_list, vector_model)
        tensor_input = torch.from_numpy(vector_input)

        test_out = net(tensor_input.float()).unsqueeze(dim=0)
        _, predict = torch.max(test_out, 1)
        if predict == 0:
            print("ほなコーンフレークちゃうかぁ...")
        else:
            print("それ、コーンフレークや！")
        i += 1
        if i == 5:
            print("もうええわ")
            break


if __name__ == "__main__":
    main()

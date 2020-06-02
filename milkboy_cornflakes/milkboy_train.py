## milkboy_train.py

import numpy as np
import pickle
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from milkboy_data import MilkBoyDataSet, nouns_extract, vectorized_word


class NNmodel(nn.Module):
    def __init__(self):
        super(NNmodel, self).__init__()
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

def train(net, train_data):
    net.train()
    running_loss = 0
    correct = 0
    total = 0
    (x_train, y_train) = train_data
    for batch_idx in range(len(train_data)):
        optimizer.zero_grad()
        feature = x_train[batch_idx].float()
        label = y_train[batch_idx].unsqueeze(dim=0)
        output = net(feature).unsqueeze(dim=0)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct += (predict == label).item()
        total += label.size(0)
    train_loss = running_loss / len(train_data)
    train_acc = correct / total
    return net, train_loss, train_acc

def valid(net, valid_data):
    net.eval()
    running_loss = 0
    correct = 0
    total = 0
    (x_valid, y_valid) = valid_data
    with torch.no_grad():
        for batch_idx in range(len(valid_data)):
            feature = x_valid[batch_idx].float()
            label = y_valid[batch_idx].unsqueeze(dim=0)
            output = net(feature).unsqueeze(dim=0)
            loss = criterion(output, label)
            running_loss += loss.item()
            _, predict = torch.max(output, 1)
            correct += (predict == label).item()
            total += label.size(0)
    val_loss = running_loss / len(valid_data)
    val_acc = correct / total
    return net, val_loss, val_acc


def main():
    # loading word2vec
    model_dir = './entity_vector/entity_vector.model.bin'
    vector_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    # set dataset
    dataset = MilkBoyDataSet.from_csv("milkboy_data.csv")
    tensor_dataset =dataset.to_model_input(vector_model)

    num_epochs = 5
    batch_size = 20
    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)

    # train & validation
    cv = 0
    best_valid_loss, best_model = 9999, None
    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(tensor_dataset)):
        print('fold {}'.format(fold_idx+1))
        net = NNmodel()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)

        train_data = tensor_dataset[train_idx]
        valid_data = tensor_dataset[valid_idx]
        for epoch_idx in range(num_epochs):
            net, train_loss, train_acc = train(net, train_data)
            net, valid_loss, valid_acc = valid(net, valid_data)
            print('train_loss {:.3f} valid_loss {:.3f} train_acc {:.3f} valid_acc {:.3f}'.format(train_loss, valid_loss, train_acc, valid_acc))

            if best_valid_loss > valid_loss:
                best_model = net
                best_valid_loss = valid_loss

    cv += valid_acc / fold.n_splits

    if best_model is not None:
        with open("milkboy_model.pickle", "wb") as f:
            pickle.dump(best_model, f)

    print("Finish Training.")
    print('cross_validate_accuracy {:.1f}%'.format(cv*100))


if __name__ == "__main__":
    main()

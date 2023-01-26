import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nltk_lib import tokenizer, stemming, bag_of_words

with open("./training data/intents.json", 'r') as stream:
    intents = json.load(stream)

words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        _tokenize = tokenizer(pattern)
        words.extend(_tokenize)
        xy.append((_tokenize, tag))
        # print(all_words)
    for response in intent['responses']:
        pass

ignore_words = ['?','!','.',',']
words = [stemming(i) for i in words if i not in ignore_words]
words = sorted(set(words))

train_x = []
train_y = []
for(tokenized_sentence, tag) in xy:
    bag = bag_of_words(tokenized_sentence, words)
    train_x.append(bag)

    label = tags.index(tag)
    train_y.append(label) # CrossEntropyloss

train_x = np.array(train_x)
train_y = np.array(train_y)

class CharDataset(Dataset):
    def __init__(self):
        self.x_samples = len(train_x)
        self.x_data = train_x
        self.y_data = train_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_samples

# Hyperparameters
batch_size = 8

dataset = CharDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=2, shuffle=True)

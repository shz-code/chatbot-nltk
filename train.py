import json
from nltk_lib import tokenizer, stemming, bag_of_words

with open("./training data/dataset.json", 'r') as stream:
    dataset = json.load(stream)

words = []
tags = []
xy = []

for data in dataset['dataset']:
    tag = data['tag']
    tags.append(tag)
    for pattern in data['patterns']:
        _tokenize = tokenizer(pattern)
        words.extend(_tokenize)
        xy.append((_tokenize, tag))
        # print(all_words)
    for response in data['responses']:
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

print(train_x)

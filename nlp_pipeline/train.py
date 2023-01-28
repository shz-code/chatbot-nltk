import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork
from nltk_lib import tokenizer, stemming, bag_of_words

with open("./nlp_pipeline/training data/intents.json", 'r') as stream:
    intents = json.load(stream)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        _tokenize = tokenizer(pattern)
        all_words.extend(_tokenize)
        xy.append((_tokenize, tag))

ignore_words = ['?','!','.',',']
all_words = [stemming(i) for i in all_words if i not in ignore_words]
all_words = sorted(set(all_words))

train_x = []
train_y = []
for(tokenized_sentence, tag) in xy:
    bag = bag_of_words(tokenized_sentence, all_words)
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
input_size = len(train_x[0])
hidden_size = 8
num_classes = len(tags)
lr = 0.001
num_epochs = 2000

dataset = CharDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_size,hidden_size,num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward
        optputs = model(words)
        loss = criterion(optputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch+1)%100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"num_classes": num_classes,
"all_words": all_words,
"tags": tags
}

FILE = "./nlp_pipeline/data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
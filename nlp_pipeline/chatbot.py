import random
import json
import torch
from nlp_pipeline.model import NeuralNetwork
from nlp_pipeline.nltk_lib import bag_of_words, tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./nlp_pipeline/training data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "./nlp_pipeline/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
num_classes = data["num_classes"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(model_state)
model.eval()


def chatbot(inp):
    sentence = tokenizer(inp)
    data = bag_of_words(sentence, all_words)
    data = data.reshape(1, data.shape[0])
    data = torch.from_numpy(data).to(device)

    output = model(data)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # print(predicted)
    # print(prob)
    if prob.item() > 0.80:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
                break
    else:
        return "I don't understand what are you saying? Maybe i'm still a dumb :("

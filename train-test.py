from curses.ascii import isdigit
import numpy as np
import random
import json
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", help = "path to the saved intents file")
    parser.add_argument("--output_dir", help = "path to save the trained model")
    args = parser.parse_args()
    return args
    

def process_data():
    all_words = []
    tags = []
    xy = []     
    args = get_args()
    intents_path = args.intents
    intents = json.load(open(intents_path))
    
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        tag = intent['tag']
        # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = tokenize(pattern)
            # add to our words list
            all_words.extend(w)
            # add to xy pair
            xy.append((w, tag))

    # stem and lower each word
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words or not w.isdigit()]
    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    return xy, all_words, tags
    
def load_data():
    xy, all_words, tags = process_data()
    # create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train

xy, all_words, tags = process_data()
X_train, y_train = load_data()
# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def train():  
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'final loss: {loss.item():.4f}')

    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }
    # save the trained model
    args = get_args()
    output_dir = args.output_dir
    FILE = "data.pth"
    torch.save(data, os.path.join(output_dir, FILE))

    print(f'training complete. file saved to {os.path.join(output_dir, FILE)}')    

if __name__ == '__main__':
    train()

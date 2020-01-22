import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Name of the data directory",
                    type=str)
parser.add_argument("arch", help="Architecture of the pre-trained model",
                    type=str)
parser.add_argument("--learning_rate", help="Learning rate of training",
                    type=float)
parser.add_argument("--epochs", help="Number of epochs",
                    type=int)
parser.add_argument("--hidden_units", help="Number of hidden units in the first classifier layer",
                    type=int)
parser.add_argument("--gpu", help="select cpu or gpu for training",
                    type=str)

parser.add_argument("--save_dir", help="Directory to save the chechpoint",
                    type=str)

args = parser.parse_args() 

data_dir=args.data_dir
arch=args.arch


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define a transform to normalize the data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(train_dir,transform=train_transforms)

# Download and load the training data

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# TODO: Define your transforms for the training, validation, and testing sets
validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
validloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


def find_class_to_idx(dir):
    indices = os.listdir(dir)
    indices.sort()
    class_to_idx = {i:indices[i] for i in range(len(indices))}
    return class_to_idx

class_to_index=find_class_to_idx(train_dir)


model = models.__dict__[arch](pretrained=True)
if arch=='vgg16':
    input_dim=model.classifier[0].in_features
elif arch=='densenet121':
    input_dim=model.classifier.in_features
    
hidden_units=args.hidden_units


#model = models.vgg16(pretrained=True)
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_dim, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1',nn.Dropout(p=.30)),  
                          ('fc2', nn.Linear(hidden_units, 1000)),
                          ('relu2', nn.ReLU()), 
                          ('drop2',nn.Dropout(p=.30)), 
                          ('fc3', nn.Linear(1000, 102)),  
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
if args.learning_rate:
    lr=args.learning_rate
else:
    lr=.001

optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
if args.gpu=='gpu':
    device='cuda'
else:
    device='cpu'
    
model.to(device)




if args.epochs:
    epochs=args.epochs
else:
    epochs = 5
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

        
        
test_loss = 0 
test_accuracy = 0 
model.eval() 
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device) 
        logps = model.forward(inputs) 
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        # Calculate accuracy 
        ps = torch.exp(logps) 
        top_p, top_class = ps.topk(1, dim=1) 
        equals = top_class == labels.view(*top_class.shape) 
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. " 
          f"Test accuracy: {test_accuracy/len(testloader):.3f}")
        
        
model.class_to_index=class_to_index


checkpoint = {'architecture': arch, 
               'model_state_dict': model.state_dict(), 
               'model_class_to_index': model.class_to_index,  
               'model_classifier': model.classifier}
if args.save_dir:
    torch.save(checkpoint, args.save_dir+'checkpoint_4b.pth')
else:
    torch.save(checkpoint, 'checkpoint_4b.pth')

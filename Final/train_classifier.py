from utils import get_dataloaders, device
from classifier import Classifier
import torch
from tqdm import tqdm

import numpy as np

import lightning.pytorch as pl

from sklearn.metrics import classification_report, accuracy_score

def train_classifier():
    train_dataloader, valid_dataloader = get_dataloaders(normalize=True)
    classifier = Classifier(lr=0.001)
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(classifier, train_dataloader, valid_dataloader)
    torch.save(classifier, 'classifier.pt')

def accuracy(classifier, dataloader):
    num_datapoints = len(dataloader)
    labels = np.zeros(num_datapoints)
    predictions = np.zeros(num_datapoints)
    for index, (x, y) in tqdm(enumerate(dataloader)):
        y_pred = classifier(x)
        y_pred = torch.argmax(y_pred, dim=1)
        labels[index] = y
        predictions[index] = y_pred.item()
    print(classification_report(labels, predictions))
    print(accuracy_score(labels,predictions))

def recon_accuracy(classifier, autoencoder, dataloader):
    classifier = classifier.to(device()).eval()
    autoencoder = autoencoder.to(device()).eval()
    num_datapoints = len(dataloader)
    labels = np.zeros(num_datapoints)
    predictions = np.zeros(num_datapoints)
    for index, (x, y) in tqdm(enumerate(dataloader)):
        x = x.to(device())
        recon, _ = autoencoder(x)
        y_pred = classifier(recon)
        y_pred = torch.argmax(y_pred, dim=1).detach().cpu()
        labels[index] = y
        predictions[index] = y_pred.item()
    print(classification_report(labels, predictions))
    print(accuracy_score(labels,predictions))

if __name__ == '__main__':
    train_dataloader, valid_dataloader = get_dataloaders(normalize=True)
    #train_classifier()    
    classifier = torch.load('classifier.pt')
    autoencoder = torch.load('model.pt')
    vqautoencoder = torch.load('vqvae_512ncw_32dim40.pt')
    accuracy(classifier, valid_dataloader)
    recon_accuracy(classifier, autoencoder, valid_dataloader)
    recon_accuracy(classifier, vqautoencoder, valid_dataloader)
        
        

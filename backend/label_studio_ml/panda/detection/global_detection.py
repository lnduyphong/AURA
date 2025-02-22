import torch 
import torch.nn.functional as F
from panda.detection.model import *
import numpy as np
from panda.detection.loss import *
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
import pandas as pd



n_epoch = 200
learning_rate = 0.001
forget_rate = 0.25
rate_schedule = np.ones(n_epoch)*forget_rate
num_gradual = 10
exponent = 3
rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(dataset, noise_indices, epoch, model1, optimizer1, model2, optimizer2, n_labels):
    clean_instances = dataset.drop(noise_indices)
    
    X_train, y_train = clean_instances.iloc[:, :-1].values, clean_instances.iloc[:, -1].values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, drop_last=True, shuffle=True)

    model1.train()
    model2.train()
    
    train_total=0
    train_correct1=0 
    train_correct2=0 
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        logits1 = model1(X_batch)
        acc1 = accuracy(logit=logits1, target=y_batch)
        
        logits2 = model2(X_batch)
        acc2 = accuracy(logit=logits2, target=y_batch)
        
        loss1, loss2 = loss_coteaching(logits1, logits2, y_batch, rate_schedule[epoch], n_labels)
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        train_total += 1
        train_correct1 += acc1[0]
        train_correct2 += acc2[0]

    return train_correct1 / train_total, train_correct2 / train_total

def correct_noise_subset(dataset, noise_indices, model1, model2, is_last=False):
    noise_instances = dataset.iloc[noise_indices]
    
    X_test, y_test = noise_instances.iloc[:, :-1].values, noise_instances.iloc[:, -1].values
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits1 = model1(X_test)
        logits2 = model2(X_test)
        
        avg_logits = (logits1 + logits2) / 2
        predicted_labels = avg_logits.argmax(axis=1).cpu()
        confidence = 1 - entropy(avg_logits.T.cpu())
        
    difference_mask = (predicted_labels != y_test)
    low_confidence_mask = confidence < np.percentile(confidence, 90)
    # if is_last == True:
    #     low_confidence_mask = confidence < 9999
    
    y_test[difference_mask & ~low_confidence_mask] = predicted_labels[difference_mask & ~low_confidence_mask]
    noise_indices = noise_instances.index[difference_mask & low_confidence_mask]
    return noise_indices, y_test
    

def run(dataset, noise_indices, n_labels, max_iters: int = 10):
    model1 = RobustNeuralNetwork(dataset.shape[1] - 1, n_labels).to(device)
    model2 = LeakyNeuralNetwork(dataset.shape[1] - 1, n_labels).to(device)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    
    for interation in range(max_iters):
        if len(noise_indices) <= 1:
            break
        train_acc1, train_acc2 = train(dataset, noise_indices, interation, model1, optimizer1, model2, optimizer2, n_labels)
        print(f"Iteration {interation}: Train Acc1: {train_acc1}, Train Acc2: {train_acc2}")
        old_noise_indices = noise_indices
        if interation == max_iters - 1:
            noise_indices, new_label = correct_noise_subset(dataset, noise_indices, model1, model2, is_last=True)
        else:
            noise_indices, new_label = correct_noise_subset(dataset, noise_indices, model1, model2)
        dataset.loc[old_noise_indices, "weak_label"] = new_label
    return dataset

if __name__ == "__main__":
    dataset = pd.read_csv("backend/label_studio_ml/panda/test/cifar_dino.csv")
    noise_indices = dataset[dataset["label"] != dataset["weak_label"]].index
    
    print("Clean rate:", np.mean(dataset["label"] == dataset["weak_label"]) * 100, "%")
    clean_labels = dataset["label"]
    
    dataset = dataset.drop(["label"], axis=1)
    dataset = run(dataset, noise_indices, max_iters=25)
    print("Clean rate:", np.mean(clean_labels == dataset.iloc[:, -1].values) * 100, "%")
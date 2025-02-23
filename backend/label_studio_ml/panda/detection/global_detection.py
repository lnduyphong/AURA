import torch 
import torch.nn.functional as F
from panda.detection import network
import numpy as np
from panda.detection import loss
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

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

def mixup(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[index]
    y_mix = lam * y + (1 - lam) * y[index]
    return X_mix, y_mix

def run(dataset, noise_indices, n_labels, num_epochs=100, patience=10, learning_rate=0.001, forget_rate=0.25, num_gradual=5, exponent=1):
    rate_schedule = np.ones(num_epochs)*forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
    
    model1 = network.RobustNeuralNetwork(dataset.shape[1] - 1, n_labels).to(device)
    model2 = network.RobustNeuralNetwork(dataset.shape[1] - 1, n_labels).to(device)
    model3 = network.RobustNeuralNetwork(dataset.shape[1] - 1, n_labels).to(device)
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
    
    clean_samples = dataset.drop(noise_indices)
    
    X_train, X_val, y_train, y_val = train_test_split(clean_samples.iloc[:, :-1].values, clean_samples.iloc[:,-1].values, test_size=0.2)
    # X_train, y_train = mixup(X_train, y_train)
    
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=64)
    
    best_val_loss = [float("inf")] * 3
    best_models = [None] * 3
    patience_counter = [0] * 3
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        model1.train()
        model2.train()
        model3.train()
        
        train_accu1, train_accu2, train_accu3 = 0, 0, 0
        val_accu1, val_accu2, val_accu3 = 0, 0, 0
        train_total, val_total = 0, 0
        train_loss1, train_loss2, train_loss3 = 0.0, 0.0, 0.0
        
        for inputs, targets in train_loader:
            if len(inputs) == 1:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits1 = model1(inputs)
            logits2 = model2(inputs)
            logits3 = model3(inputs)
            
            loss1, loss2, loss3 = loss.loss_coteaching(logits1, logits2, logits3, targets, rate_schedule[epoch], n_labels)
            
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()
            
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            
            train_accu1 += accuracy(logits1, targets)[0]
            train_accu2 += accuracy(logits2, targets)[0]
            train_accu3 += accuracy(logits3, targets)[0]
            train_total += 1
            
        train_loss1 /= train_total
        train_loss2 /= train_total
        train_loss3 /= train_total
        print(f"Train Loss: [{train_loss1:.4f}, {train_loss2:.4f}, {train_loss3:.4f}], Train: [{(train_accu1/train_total).item():.3f}, {(train_accu2/train_total).item():.3f}, {(train_accu3/train_total).item():.3f}]")
        model1.eval()
        model2.eval()
        model3.eval()
        
        val_losses = [0.0] * 3
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                logits1 = model1(inputs)
                logits2 = model2(inputs)
                logits3 = model3(inputs)
                
                val_loss1 = F.cross_entropy(logits1, targets, reduction='none')
                val_loss2 = F.cross_entropy(logits2, targets, reduction='none')
                val_loss3 = F.cross_entropy(logits3, targets, reduction='none')
                
                val_losses[0] += val_loss1.mean().item()
                val_losses[1] += val_loss2.mean().item()
                val_losses[2] += val_loss3.mean().item()
                
                val_accu1 += accuracy(logits1, targets)[0]
                val_accu2 += accuracy(logits2, targets)[0]
                val_accu3 += accuracy(logits3, targets)[0]
                val_total += 1
        
        val_losses = [loss / len(val_loader) for loss in val_losses]
        print(f"Val Losses: [{val_losses[0]:.4f}, {val_losses[1]:.4f}, {val_losses[2]:.4f}], Validation: [{(val_accu1/val_total).item():.3f}, {(val_accu2/val_total).item():.3f}, {(val_accu3/val_total).item():.3f}]")
        
        for i, (val_loss, model) in enumerate(zip(val_losses, [model1, model2, model3])):
            if val_loss < best_val_loss[i]:
                best_val_loss[i] = val_loss
                best_models[i] = model.state_dict()
                patience_counter[i] = 0 
            else:
                patience_counter[i] += 1      
                
        if all(count >= patience for count in patience_counter):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load the best models
    model1.load_state_dict(best_models[0])
    model2.load_state_dict(best_models[1])
    model3.load_state_dict(best_models[2])
    
    logits1 = model1(torch.Tensor(dataset.iloc[:, :-1].values).to(device)).detach().cpu().numpy()
    logits2 = model2(torch.Tensor(dataset.iloc[:, :-1].values).to(device)).detach().cpu().numpy()
    logits3 = model3(torch.Tensor(dataset.iloc[:, :-1].values).to(device)).detach().cpu().numpy()
    
    return logits1, logits2, logits3

if __name__ == "__main__":
    dataset = pd.read_csv("backend/label_studio_ml/panda/test/cifar3k.csv")
    noise_indices = dataset[dataset["label"] != dataset["weak_label"]].index
    
    print("Clean rate:", np.mean(dataset["label"] == dataset["weak_label"]) * 100, "%")
    clean_labels = dataset["label"]
    
    dataset = dataset.drop(["label"], axis=1)
    dataset = run(dataset, noise_indices, n_labels=10, num_epochs=100, patience=5, learning_rate=0.001, forget_rate=0.25, num_gradual=5, exponent=1)
    # print("Clean rate:", np.mean(clean_labels == dataset.iloc[:, -1].values) * 100, "%")
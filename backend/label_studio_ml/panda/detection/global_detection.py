import torch

def global_detection(noise_instances, clean_instances, iteration: int):
    clean_features = clean_instances.iloc[:, :-1].values
    clean_labels = clean_instances.iloc[:, -1].values
    
    noise_features = noise_instances.iloc[:, :-1].values
    noise_labels = noise_instances.iloc[:, -1].values
    
    X_test = torch.tensor(noise_features, dtype=torch.float32)
    y_test = noise_labels
    
    model = self.initiate_neural_net(clean_features, clean_labels)

    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(self.device))
        pred_labels = pred.argmax(dim=1).cpu().numpy()

    # true_label = pd.Series(raw_labels).iloc[self.noise_indices]
        
    # for i in range(len(pred_labels)):
    #     print("Pred:", pred_labels[i].tolist(), "Refined:", y_test[i].tolist(), "TrueLabel:", true_label.iloc[i], "Result:", y_test[i]==pred_labels[i])
    
    noise_indices = noise_indices[(pred_labels != y_test)]
    
    return noise_indices
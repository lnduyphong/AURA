import os
import shutil
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_labels, model_saving_path):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(128, 32)
        self.dropout5 = nn.Dropout(0.3)
        self.fc6 = nn.Linear(32, n_labels)
        self.model_saving_path = model_saving_path

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)
        x = torch.softmax(self.fc6(x), dim=1)
        return x
    
    def delete_weights(self):
        try:
            if os.path.exists(self.model_saving_path):
                shutil.rmtree(self.model_saving_path)
                print(f"Folder {self.model_saving_path} has been deleted successfully.")
        except Exception as e:
            print(f"Can not remove folder: {e}")

    def initiate_neural_net(self, X, y, n_labels):
        model_saving_dir = f"label-studio-ml-backend/my_ml_backend/model_saving/{str(time.time())}"
        if not os.path.exists(model_saving_dir):
            os.makedirs(model_saving_dir)
        model_saving_path = os.path.join(model_saving_dir, f'best_model.pt')
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        model = NeuralNetwork(X.shape[1], n_labels, model_saving_path).to(self.device)
        best_val_loss = float('inf')
        patience = 10
        early_stopping_counter = 0
        optimizer = optim.AdamW(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        epochs = 200
        batch_size = 256
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size)
        
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(device))
                val_loss = criterion(val_outputs, y_val.to(device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                try:
                    torch.save(model.state_dict(), model_saving_path)
                except Exception as e:
                    print(f"Cannot save the model: {e}")
            else:
                early_stopping_counter += 1
                scheduler.step(val_loss)
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            if epoch % 30 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}")

        try:
            model.load_state_dict(torch.load(model_saving_path, weights_only=True))
        except:
            print('Failed to load the best model.')
            
        try:
            if os.path.exists(model_saving_dir):
                shutil.rmtree(model_saving_dir)
                print(f"Folder {model_saving_dir} has been deleted successfully.")
        except Exception as e:
            print(f"Can not remove folder: {e}")
        return model
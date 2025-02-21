import os
import torch
import numpy as np
import pandas as pd
import shutil
import time
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from backend.label_studio_ml.panda.model.NeuralNetwork import NeuralNetwork

class DeCoTool:
    def __init__(self, original_data, corrupted_label_index, n_labels, true_labels, dataset_name = None, encode_model = None):
        self.original_data = original_data
        self.true_labels = true_labels
        self.n_instances = len(original_data) # ???????
        self.n_labels = n_labels
        self.dataset_name = dataset_name # cut
        self.encode_model = encode_model # cut
        self.corrupted_label_index = corrupted_label_index # kho hieu vl
        self.n_corrupted_labels = len(corrupted_label_index) # cut
        self.noise_indices = pd.Index([]) # cut
        self.noise_pattern = None # maybe cut
        self.refined_indices = None # label da sua, cut
        self.injection_model = None # maybe cut
        self.fixed_labels = None # cut
        self.embedding_layer = None # ?
        self.refine_model = None # cut
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # global nen cut
        print('Device:', self.device)
        
    def local_detection(self, k_neighbors=2, alpha=0.1):
        X_features = self.original_data.iloc[:, :-1].values
        y = self.original_data.iloc[:, -1].values

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors+1)
        knn_model.fit(X_features, y)

        distances, neighbor_indices = knn_model.kneighbors(X_features)

        filtered_indices = neighbor_indices[:, 1:k_neighbors+1]
        filtered_distances = distances[:, 1:k_neighbors+1]

        predicted_labels = []
        prediction_probabilities = []

        for i in range(len(filtered_indices)):
            neighbor_labels = y[filtered_indices[i]]
            neighbor_distances = filtered_distances[i]

            weight_sum = np.sum(1 / (neighbor_distances ** 2 + 1e-10))
            weighted_votes = np.zeros(self.n_labels)

            for j in range(len(neighbor_labels)):
                weighted_votes[neighbor_labels[j]] += 1 / (neighbor_distances[j] ** 2 + 1e-10)

            weighted_prob = weighted_votes / weight_sum
            prediction_probabilities.append(weighted_prob)
            predicted_labels.append(np.argmax(weighted_prob))

        prediction_probabilities = np.array(prediction_probabilities)
        predicted_labels = np.array(predicted_labels)

        def cosine_similarity(index, probability_vector):
            one_hot_label = np.zeros(len(probability_vector))
            one_hot_label[y[index]] = 1

            dot_product = np.dot(probability_vector, one_hot_label)
            norm_prob = np.linalg.norm(probability_vector)
            norm_label = np.linalg.norm(one_hot_label)

            return np.abs(dot_product / (norm_prob * norm_label))

        def calculate_threshold(pred_labels, true_labels, category, category_weight):
            num_predicted = np.sum(pred_labels == category)
            num_consensus = np.sum((pred_labels == category) & (true_labels == category))
            threshold_value = 1 - num_consensus / num_predicted + alpha * category_weight
            return min(max(threshold_value, 0.01), 1)

        credibility_scores = []
        for index, prob_vector in enumerate(prediction_probabilities):
            credit_score = cosine_similarity(index, prob_vector)
            credibility_scores.append((self.original_data.index[index], predicted_labels[index], credit_score))

        sorted_credibility_scores = sorted(credibility_scores, key=lambda x: x[2])

        for category in tqdm(range(self.n_labels)):
            instances_in_category = [sample for sample in sorted_credibility_scores if sample[1] == category]
            category_weight = len(instances_in_category) / (self.n_instances / self.n_labels)
            threshold_value = calculate_threshold(predicted_labels, y, category, category_weight)
            noise_limit = int(threshold_value * len(instances_in_category))
            potential_noise_index = pd.Index([instances_in_category[i][0] for i in range(noise_limit)])
            if self.noise_indices is None or len(self.noise_indices) == 0:
                self.noise_indices = potential_noise_index
            else:
                self.noise_indices = pd.Index(pd.concat([pd.Series(self.noise_indices), pd.Series(potential_noise_index)]).unique())

        self.noise_instances = self.original_data.iloc[self.noise_indices]
        self.clean_instances = self.original_data.drop(self.noise_indices)
        
        self.print_results(phase='local')

    def initiate_neural_net(self, X, y):
        model_saving_dir = f"label-studio-ml-backend/my_ml_backend/model_saving/{str(time.time())}"
        if not os.path.exists(model_saving_dir):
            os.makedirs(model_saving_dir)
        model_saving_path = os.path.join(model_saving_dir, f'best_model.pt')
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        model = NeuralNetwork(X.shape[1], self.n_labels, model_saving_path).to(self.device)
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
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(self.device))
                val_loss = criterion(val_outputs, y_val.to(self.device)).item()
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

    def global_detection(self, iteration: int):
        self.clean_instances = self.original_data.drop(self.noise_indices)
        clean_features = self.clean_instances.iloc[:, :-1].values
        clean_labels = self.clean_instances.iloc[:, -1].values
        
        self.noise_instances = self.original_data.iloc[self.noise_indices]
        noise_features = self.noise_instances.iloc[:, :-1].values
        noise_labels = self.noise_instances.iloc[:, -1].values
        
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
        
        self.noise_indices = self.noise_indices[(pred_labels != y_test)]
        
        self.print_results(phase='global', iteration=iteration)
        
    def print_results(self, phase, iteration=0):
        if self.refined_indices is None:
            total_noise = self.noise_indices
        else:
            total_noise = pd.Index(pd.concat([pd.Series(self.noise_indices), pd.Series(self.refined_indices)]).unique())
        correct_detection_indices = [index for index in total_noise if index in self.corrupted_label_index]
        wrong_detection_indices = [index for index in total_noise if index not in self.corrupted_label_index]

        precision = 0 if len(total_noise) == 0 else len(correct_detection_indices) / len(total_noise)
        recall = 0 if self.n_corrupted_labels == 0 else len(correct_detection_indices) / self.n_corrupted_labels
        F1_score = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        print('--------------------------------------------')
        if phase == 'global':
            print('Global Results:')
        else:
            print('Local Results:')
        print(f'Iteration: {iteration + 1}, Dataset: {self.dataset_name}')
        print(f'Precision: {round(precision, 3)}')
        print(f'Recall: {round(recall, 3)}')
        print(f'F1: {round(F1_score, 3)}')
        
        print(f"# of ~ clean set: {(self.n_instances - len(self.noise_indices))}")
        print(f"# of noise in ~ clean set: {(len(self.corrupted_label_index) - len(correct_detection_indices))}")
        print(f'# of current noisy instances: {len(self.noise_indices)}')
        print(f'# of total noisy instances: {len(total_noise)}')
        print(f'# of wrong detected error instances: {len(wrong_detection_indices)}')
        print(f'# of true detected error instances: {len(correct_detection_indices)}')
        print('--------------------------------------------\n\n') 

    def refine_noise(self):

        X_train = torch.tensor(self.clean_instances.iloc[:, :-1].values, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.clean_instances.iloc[:, -1].values, dtype=torch.long).to(self.device)
    
        noise_labels = torch.tensor(self.noise_pattern, dtype=torch.long).to(self.device)
    
        embedding_dim = 10
        if self.embedding_layer is None:
            self.embedding_layer = torch.nn.Embedding(self.n_labels, embedding_dim).to(self.device)
        y_noise_embed = self.embedding_layer(noise_labels)
    
        alpha = 0.5
        X_train = torch.cat((X_train, alpha * y_noise_embed), dim=1)
    
        if self.refine_model is None:
            self.refine_model = self.initiate_neural_net(X_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    
        self.refine_model.eval()
        with torch.no_grad():
            noise_labels = torch.tensor(self.noise_instances.iloc[:, -1].values, dtype=torch.long).to(self.device)
            y_noise_embed = self.embedding_layer(noise_labels)
    
            X_fix = torch.tensor(self.noise_instances.iloc[:, :-1].values, dtype=torch.float32).to(self.device)
            X_fix = torch.cat((X_fix, alpha * y_noise_embed), dim=1)
    
            probs = self.refine_model(X_fix)
            self.fixed_labels = probs.argmax(dim=1).cpu().numpy()
    
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            self.fixing_confidence_scores = (1 - entropy / torch.log(torch.tensor(probs.shape[1]))).cpu().numpy()

        # confidence_threshold = np.percentile(self.fixing_confidence_scores, 70)
        confidence_threshold = -9999999
        
        high_confidence_indices = self.noise_indices[self.fixing_confidence_scores >= confidence_threshold]
        
        values_to_assign = np.array(self.fixed_labels[self.fixing_confidence_scores >= confidence_threshold], dtype=self.original_data.iloc[:, -1].dtype)
        self.original_data.iloc[high_confidence_indices, -1] = values_to_assign
        
        if self.refined_indices is None or len(self.refined_indices) == 0:
            self.refined_indices = high_confidence_indices
        else:
            self.refined_indices = pd.Index(pd.concat([pd.Series(self.refined_indices), pd.Series(high_confidence_indices)]).unique())
            # print(f"\n\n--------------------------------------------")
            # print(f"# of noise instance:", len(y_test))
            # print(f"# of refined instances:", len(high_confidence_indices))
            # print(f"# of correct fixing instances:", np.sum(y_test == true_label))
            # print(f"# of correct pred instances:", np.sum(pred_labels == true_label))
            # print(f"--------------------------------------------\n\n")
        print(f"Purity of dataset:", round(1 - np.mean(self.original_data.iloc[:, -1].values != self.true_labels), 3))
        print(f"Correction rate:", round(1 - np.mean(self.original_data.iloc[self.corrupted_label_index, -1].values != pd.Series(self.true_labels).iloc[self.corrupted_label_index].values.flatten()), 3))

    def inject_noise(self):
        print("Applying noise pattern to clean dataset")
        self.clean_instances = self.original_data.drop(self.noise_indices)
        self.noise_instances = self.original_data.iloc[self.noise_indices]
        
        
        if self.injection_model is None:
            self.injection_model = self.initiate_neural_net(self.noise_instances.iloc[:, :-1].values, self.noise_instances.iloc[:, -1].values.flatten())
        
        self.injection_model.eval()
        with torch.no_grad():
            pred = self.injection_model(torch.tensor(self.clean_instances.iloc[:, :-1].values, dtype=torch.float32).to(self.device))
            self.noise_pattern = pred.argmax(dim=1).cpu().numpy()
            
        unique_labels, counts = torch.unique(torch.tensor(self.noise_pattern), return_counts=True)
        print(f"Noise label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

        unique_labels, counts = torch.unique(torch.tensor(self.clean_instances.iloc[:,-1].values), return_counts=True)
        print(f"Clean label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

        print(np.mean(self.clean_instances.iloc[:,-1].values == self.noise_pattern))
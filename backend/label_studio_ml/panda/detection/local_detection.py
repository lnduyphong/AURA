import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

alpha = 0.0

def cosine_similarity(index, y, probability_vector):
    one_hot_label = np.zeros(len(probability_vector))
    one_hot_label[y[index]] = 1

    dot_product = np.dot(probability_vector, one_hot_label)
    norm_prob = np.linalg.norm(probability_vector)
    norm_label = np.linalg.norm(one_hot_label)

    return np.abs(dot_product / (norm_prob * norm_label))

def calculate_threshold(pred_labels, current_labels, category, category_weight):
    num_predicted = np.sum(pred_labels == category)
    # print(f"num predicted: {num_predicted}")
    # print(f"preds {pred_labels}")
    # print(f"category: {category}")
    num_consensus = np.sum((pred_labels == category) & (current_labels == category))
    threshold_value = 1 - num_consensus / (num_predicted + 1e-10) + alpha * category_weight
    return min(max(threshold_value, 0.01), 1)

def run(data, n_labels, k_neighbors=10):
    n_instances = len(data)
    y = data['weak_label'].values
    X_features = data.drop(['weak_label'], axis=1).values
    knn_model= KNeighborsClassifier(n_neighbors=k_neighbors + 1)
    knn_model.fit(X_features, y)

    distances, neighbor_indices = knn_model.kneighbors(X_features)

    filtered_indices = neighbor_indices [:, 1:k_neighbors+1]
    filtered_distances = distances[:, 1:k_neighbors+1]

    preds_labels = []
    preds_probs = []


    for i in range(len(filtered_indices)):
        neighbor_labels = y[filtered_indices[i]]
        neighbor_distances = filtered_distances[i]

        weight_sum = np.sum(1 / (neighbor_distances ** 2 + 1e-10))
        weighted_votes = np.zeros(n_labels)

        for j in range(len(neighbor_labels)):
            weighted_votes[neighbor_labels[j]] += 1 / (neighbor_distances[j] + 1e-10)

        weighted_prob = weighted_votes / weight_sum
        preds_probs.append(weighted_prob)
        preds_labels.append(np.argmax(weighted_prob))

    preds_probs = np.array(preds_probs)
    preds_labels = np.array(preds_labels)

    noise_indices = data[preds_labels == data.iloc[:, -1].values].index

    return noise_indices

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

alpha = 0.1

def cosine_similarity(index, y, probability_vector):
    one_hot_label = np.zeros(len(probability_vector))
    one_hot_label[y[index]] = 1

    dot_product = np.dot(probability_vector, one_hot_label)
    norm_prob = np.linalg.norm(probability_vector)
    norm_label = np.linalg.norm(one_hot_label)

    return np.abs(dot_product / (norm_prob * norm_label))

def calculate_threshold(pred_labels, current_labels, category, category_weight):
    num_predicted = np.sum(pred_labels == category)
    print(f"num predicted: {num_predicted}")
    print(f"preds {pred_labels}")
    print(f"category: {category}")
    num_consensus = np.sum((pred_labels == category) & (current_labels == category))
    threshold_value = 1 - num_consensus / (num_predicted + 1e-10) + alpha * category_weight
    return min(max(threshold_value, 0.01), 1)

def run(data, n_labels, k_neighbors=2):
    n_instances = len(data)
    y = data['weak_label'].values
    X_features = data.drop(['weak_label'], axis=1).values
    print(X_features)
    print(y)
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
            print(neighbor_distances[j])
            weighted_votes[neighbor_labels[j]] += 1 / (neighbor_distances[j] ** 2 + 1e-10)

        weighted_prob = weighted_votes / weight_sum
        preds_probs.append(weighted_prob)
        preds_labels.append(np.argmax(weighted_prob))

    preds_probs = np.array(preds_probs)
    preds_labels = np.array(preds_labels)

    credibility_scores = []
    for index, prob_vector in enumerate(preds_probs):
        credit_score = cosine_similarity(index, y, prob_vector)
        credibility_scores.append((data.index[index], preds_labels[index], credit_score))

    sorted_credibility_scores = sorted(credibility_scores, key=lambda x: x[2])

    noise_indices = None
    for category in tqdm(range(n_labels)):
        instances_in_category = [sample for sample in sorted_credibility_scores if sample[1] == category]
        category_weight = len(instances_in_category) / (n_instances / n_labels)
        threshold_value = calculate_threshold(preds_labels, y, category, category_weight)
        noise_limit = int(threshold_value * len(instances_in_category))
        potential_noise_index = pd.Index([instances_in_category[i][0] for i in range(noise_limit)])
        if noise_indices is None or len(noise_indices) == 0:
            noise_indices = potential_noise_index
        else:
            noise_indices = pd.Index(pd.concat([pd.Series(noise_indices), pd.Series(potential_noise_index)]).unique())

    noise_instances = data.iloc[noise_indices]
    clean_instances = data.drop(noise_indices, axis=0)

    return noise_instances, clean_instances, noise_indices

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

def run(original_data, n_labels, k_neighbors=2, alpha=0.1):
    n_instances = len(original_data)
    y_raw = original_data['weak_label'].values
    print(original_data)
    X_features = original_data.drop(['weak_label'], axis=1).values
    label_mapping = {label: idx for idx, label in enumerate(np.unique(y_raw))}
    y = np.array([label_mapping[label] for label in y_raw])
    # print(y)
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
        weighted_votes = np.zeros(n_labels)

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
        credibility_scores.append((original_data.index[index], predicted_labels[index], credit_score))

    sorted_credibility_scores = sorted(credibility_scores, key=lambda x: x[2])
    print(f"sort {sorted_credibility_scores}")
    for category in tqdm(range(n_labels)):
        instances_in_category = [sample for sample in sorted_credibility_scores if sample[1] == category]
        print(f"ins: {instances_in_category}")
        category_weight = len(instances_in_category) / (n_instances / n_labels)
        print(f"w: {category_weight}")
        threshold_value = calculate_threshold(predicted_labels, y, category, category_weight)
        print(f"threshold {threshold_value}")
        noise_limit = int(threshold_value * len(instances_in_category))
        potential_noise_index = pd.Index([instances_in_category[i][0] for i in range(noise_limit)])
        noise_indices = None
        if noise_indices is None or len(noise_indices) == 0:
            noise_indices = potential_noise_index
        else:
            noise_indices = pd.Index(pd.concat([pd.Series(noise_indices), pd.Series(potential_noise_index)]).unique())

    noise_instances = original_data.iloc[noise_indices]
    clean_instances = original_data.drop(noise_indices)
    
    return noise_instances, clean_instances
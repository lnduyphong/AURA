from panda.data_processing.data import NewData 
from panda.data_processing.feature_extraction import extract_text_feature
from panda.detection import local_detection
from panda.detection import global_detection

import pandas as pd
import numpy as np


def run():
    data_df = pd.read_csv("backend/label_studio_ml/panda/test/agnews3k2.csv")
    labels = np.unique(data_df["label"].values)
    n_labels = len(labels)
    type_data = "text"
    
    clean_labels = data_df["label"]
    llm_labels = data_df["weak_label"].values.copy()
    noise_indices = data_df[data_df["label"] != data_df["weak_label"]].index
    print("Clean rate:", np.mean(data_df["label"] == data_df["weak_label"]) * 100, "%")
    data_df.drop(columns=['label', 'weak_label'], inplace=True)

    embed_vt = extract_text_feature(dataset=data_df, batch_size=256, encode_model="facebook/bart-base")
    raw_data = pd.DataFrame(embed_vt)
    # raw_data = data_df
    
    
    raw_data['weak_label'] = llm_labels
    # detected_noise_indices = local_detection.run(raw_data, n_labels, k_neighbors=13)
    detected_noise_indices = noise_indices
    print_results(noise_indices, detected_noise_indices)
    
    fixed_data = global_detection.run(raw_data.copy(), detected_noise_indices, n_labels, max_iters=50)
    print_results(noise_indices, detected_noise_indices, raw_data, fixed_data, clean_labels)

def print_results(noise_indices, detected_noise_indices, raw_data=None, fixed_data=None, clean_labels=None):
    correct_detection_indices = [index for index in detected_noise_indices if index in noise_indices]
    wrong_detection_indices = [index for index in detected_noise_indices if index not in noise_indices]
    
    precision = len(correct_detection_indices) / len(detected_noise_indices) if len(detected_noise_indices) > 0 else 1.0
    recall = len(correct_detection_indices) / len(noise_indices) if len(noise_indices) > 0 else 1.0
    F1 = 2 * precision * recall / (precision + recall) if recall + precision > 0 else 0
    
    print("--------------------------------------------")
    print(f'Precision: {round(precision, 3)}')
    print(f'Recall: {round(recall, 3)}')
    print(f'F1: {round(F1, 3)}')
    print("--------------------------------------------")
    print(f'Total noise instances: {len(noise_indices)}')
    print(f'Wrongly detected noise: {len(wrong_detection_indices)}')
    print(f'Correctly detected noise: {len(correct_detection_indices)}')
    print("--------------------------------------------")
    if raw_data is not None:
        print("Correct annotation rate:", np.mean(clean_labels == fixed_data.iloc[:, -1].values) * 100, "%")
        print("Repaired samples:", np.sum(fixed_data.iloc[:, -1].values != raw_data.iloc[:, -1].values))
        print("Correct repaired samples:", np.sum((fixed_data.iloc[:, -1].values != raw_data.iloc[:, -1].values) & (clean_labels == fixed_data.iloc[:, -1].values)))
        print("--------------------------------------------")
    
    
    
if __name__ == "__main__":
    run()
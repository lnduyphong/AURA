import numpy as np
from deCo import DeCoTool
from FeatureExtraction import *


def run(data):
    clean_labels = data["label"].values.flatten()
    n_labels = data["label"].nunique()
    data.drop("label", axis=1, inplace=True)
    
    corrupted_label_index = data[data["weak_label"] != clean_labels].index
    print("Error rate:", np.mean(data["weak_label"] != clean_labels))
    
    embedded_data, data_dir = extract_text_feature(dataset=data, batch_size=256, encode_model="facebook/bart-base")
    
    tools = DeCoTool(original_data=embedded_data, corrupted_label_index=corrupted_label_index, n_labels=n_labels, true_labels=clean_labels)
    tools.local_detection()
    for i in range(10):
        tools.global_detection(i)
    tools.inject_noise()
    tools.refine_noise()
    
    
if __name__ == "__main__":
    run(data = pd.read_csv("/home/trucddx/NCKH2025/Autonomous-Label-Studio/label-studio-ml-backend/my_ml_backend/Data/anotated-data/sarcasm-3k.csv"))
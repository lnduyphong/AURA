import numpy as np # linear algebra
import pandas as pd 
from panda.labeling.agents.chatGPT_text import agent as chatGPT_text_agent
from panda.labeling.agents.chatGPT_image import agent as chatGPT_image_agent
# from agents.deepseek_text import agent as deepseek_text_agent
# from agents.deepseek_image import agent as deepseek_image_agent
from tqdm import tqdm

def run(dataset, model_name, labels):
    # dataset = pd.read_csv(dataset).head(20)
    print(labels)
    chatGPT = chatGPT_text_agent(model_name, labels)
    final_labels = []
    characteristics = chatGPT.find_information()
    print(characteristics)
    for i in tqdm(range(len(dataset['text']))):
        text = dataset['text'][i]
        # image = "./panda/cifar10/" + image
        label, explanation = chatGPT.annotator_task(text, characteristics)
        final_labels.append(label)
        #Challenger critiques the annotation
        # print(label)
        # critique = chatGPT.challenger_task(text, label, explanation)
        # print(critique)s
        #Adjudicator makes the final decision
        # final_decision = chatGPT.adjudicator_task(text, label, explanation, critique)
        # final_labels.append(final_decision)

    # data = dataset[['label', 'text']]
    # data['weak_label'] = final_labels
    
    # numerical_label = {"Sarcasm": 1, "Non-sarcasm": 0}
    # data['weak_label'] = data['weak_label'].map(numerical_label)
    # print((data['weak_label']  == data['label']).sum())
    # data.to_csv("/home/trucddx/NCKH2025/Autonomous-Label-Studio/label-studio-ml-backend/my_ml_backend/Data/anotated-data/sarcasm.csv", index=False)
    # print(data)
    return final_labels

# run_annotation_process('/home/trucddx/NCKH2025/Autonomous-Label-Studio/label-studio-ml-backend/my_ml_backend/Data/agnews (2).csv', 'Ag-news', 'gpt-4o', ['Sci/Tech', 'Sports', 'Business', 'World'])

# run('/home/trucddx/NCKH2025/Autonomous-Label-Studio/label-studio-ml-backend/my_ml_backend/Data/sarcasm-3k (2).csv', 'sarcasm', 'gpt-4o', ['Sarcasm', 'Non-Sarcasm'])

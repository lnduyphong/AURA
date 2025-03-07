import os
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoImageProcessor
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_feature(dataset, batch_size, encode_model):

    def embed_text_batch(batch_texts, tokenizer, model):
        
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
      
    print(f"Using device: {device}")
        
    tokenizer = AutoTokenizer.from_pretrained(encode_model)
    model = AutoModel.from_pretrained(encode_model, trust_remote_code=True).to(device)
    
    embeddings = []
    labels = []

    texts = [example for example in dataset["text"]]
    # batch_labels = [example for example in dataset["weak_label"]]
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_text_batch(batch_texts, tokenizer, model)
        embeddings.append(batch_embeddings)
        # labels.extend(batch_labels[i:i + batch_size])

    embeddings = np.vstack(embeddings)
    # labels = np.array(labels)

    # dataset_dir = time.time()
    
    # embedded_data = pd.DataFrame(embeddings)
    # labels = labels.flatten()
    # embedded_data['weak_label'] = labels
    
    return embeddings

def extract_image_feature(dataset, batch_size, encode_model):
    def embed_img_batch(batch_imgs, processor, model):
        inputs = processor(batch_imgs, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    print(f"Using device: {device}")
    processor = AutoImageProcessor.from_pretrained(encode_model, do_rescale=False)
    model = AutoModel.from_pretrained(encode_model).to(device)

    embeddings = []
    data_loader = data.DataLoader(dataset, batch_size=batch_size)
    for imgs, _ in tqdm(data_loader):
        batch_embeddings = embed_img_batch(imgs, processor, model)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    return embeddings
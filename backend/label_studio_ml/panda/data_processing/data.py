import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class NewData:
    def __init__(self, tasks: List[Dict], parsed_label_config: Dict):
        self.tasks = tasks
        self.parsed_label_config = parsed_label_config

    def create_df(self):
        data_list = []
        for task in self.tasks:
            data_entry = {}
            data = task.get('data', {})
            
            if 'text' in data:
                data_entry['type'] = 'text'
                data_entry['text'] = data['text']
            elif 'image' in data:
                data_entry['type'] = 'image'
                data_entry['image_path'] = data['image']
            else:
                data_entry['type'] = 'unknown'

            if 'label' in data:
                data_entry['label'] = data['label']
            
            data_list.append(data_entry)
        
        return pd.DataFrame(data_list)
    
    def get_predictions(self, task: Dict):
        predictions = task.get('predictions', [])
        if predictions:
            for pred in predictions:
                results = pred.get('result', [])
                for res in results:
                    if res.get('type') == 'choices':
                        return res['value'].get('choices', [None])[0]  # Return first choice
        return None
    
    def get_labels(self):
        for config in self.parsed_label_config.values():
            res = config['labels']
            return res
        return None
    
    def get_type(self):
        for config in self.parsed_label_config.values():
            if config['type'] == 'Choices' and any(inp['type'] == 'Text' for inp in config['inputs']):
                return 'text'
            elif config['type'] == 'Choices' and any(inp['type'] == 'Image' for inp in config['inputs']):
                return 'image'
        return 'unknown'

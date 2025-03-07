from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from panda.data_processing.data import NewData 
from panda.data_processing.feature_extraction import extract_text_feature
from panda.detection import local_detection
from panda.detection import global_detection
from panda.labeling import controller
from panda.correction import inject_noise

# from response import ModelResponse


from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


def create_data(tasks: List[Dict], label_config):
    data = NewData(tasks, label_config)
    data_df = data.create_df()
    labels = data.get_labels()
    type_data = data.get_type()
    logger.debug(f"Prediction: {data_df}")

    # print(data_df['id'])
    # return data_df

    labeled_data = controller.run(dataset=data_df, model_name='gpt-4o', labels=labels)
    if type_data == "text":
        embed_vt = extract_text_feature(dataset=data_df, batch_size=256, encode_model="facebook/bart-base")
    else:
        embed_vt = extract_image_feature(dataset=data_df, batch_size=256, encode_model="facebook/dinov2-base")

    raw_data = pd.DataFrame(embed_vt)
    
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labeled_data))}
    map_labels = np.array([label_mapping[label] for label in labeled_data])
    raw_data['weak_label'] = map_labels   
    raw_data['label'] = data_df['label']   
    
    return raw_data, labels, type_data

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

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Run inference and return predictions in Label Studio JSON format.

        :param tasks: List of Label Studio tasks in JSON format.
        :param context: Label Studio context in JSON format.
        :return: ModelResponse containing predictions.
        """
        print(f'''\
        Running prediction on {len(tasks)} tasks
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Extra params: {self.extra_params}''')

        # Extract data from tasks and process it
        data, labels, type_data = create_data(tasks, self.parsed_label_config)
        n_labels = len(labels)
        
        print(f"Extracted {len(data)} data points with {n_labels} labels: {labels}")

        # Identify and correct noisy labels
        noise_indices = data[data["label"] != data["weak_label"]].index
        detected_noise_indices = local_detection.run(data.copy(), n_labels, k_neighbors=3)

        print_results(noise_indices, detected_noise_indices)

        fixed_data = global_detection.run(
            data.copy(), detected_noise_indices, n_labels, 
            max_epochs=100, percentile=5, actual_noise_indices=noise_indices
        )

        # Convert weak labels (integers) to label names
        results = []
        for i in range(fixed_data.shape[0]):
            label_index = int(fixed_data['weak_label'][i])  # Ensure integer type
            label = labels[label_index]  # Map index to label

            prediction_result = {
                "model_version": self.get("model_version"),
                "result": [
                    {
                        "from_name": "sentiment",
                        "id": str(i),  
                        "to_name": "text",
                        "type": "choices",
                        "value": {
                            "choices": [label]
                        }
                    }
                ],
                "score": 0  # Update if score computation is available
            }
            results.append(prediction_result)

        print(f"Generated {len(results)} predictions")
        
        return ModelResponse(predictions=results)

    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

tasks = [
   
]
parsed_label_config = {
    "sentiment": {
        "type": "Choices",
        "to_name": ["text"],
        "inputs": [{"type": "Text", "value": "text"}],
        "labels": ["Positive", "Negative", "Neutral"]
    }
}


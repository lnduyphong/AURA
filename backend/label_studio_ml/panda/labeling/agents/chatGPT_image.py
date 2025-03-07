import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import json
from openai import OpenAI
import pandas as pd
import json
import base64


OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)
class agent:
    def __init__(self, model_name, labels):
        self.model_name = model_name
        self.labels = labels

    def query(self, prompt, image_path):
        
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
    
        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        }
                    ],
                }
            ],
            response_format={ 
                "type": "json_object"
             }    
        )
        # time.sleep(0.5)
        return response.choices[0].message.content

    def query_text(self, prompt, content):
        response = client.chat.completions.create(
        model=self.model_name,
        temperature=0.1,
        # This is to enable JSON mode, making sure responses are valid json objects
        response_format={ 
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
        )
        # time.sleep(0.5)
        return response.choices[0].message.content
    
    def find_information(self):
        prompt = f"""
            Your task is to extract and provide structured information for each dataset label.  
            Focus on identifying key **visual** and **textual** attributes that define each label.  

            **Requirements:**  
            - Identify and extract **keywords** related to each label.  
            - Describe the **key characteristics** based on image representation.  
            - Specify important **properties** (such as color, shape, texture if image-based.  
            - Ensure that the descriptions are precise, factual, and relevant.  
            You will be provided with a text, and you will output a JSON object containing the following information:
            **Format:**  
            {{
                "Characteristics": {{
                    "<label_1>": {{
                        "Keywords": "<comma-separated keywords>",
                        "Description": "<detailed and accurate description>",
                        "Properties": "<specific visual or textual properties>"
                    }},
                    "<label_2>": {{
                        "Keywords": "<comma-separated keywords>",
                        "Description": "<detailed and accurate description>",
                        "Properties": "<specific visual or textual properties>"
                    }},
                    ...
                }}
            }}

            The labels that require descriptions are: {self.labels}.  
            Ensure that each entry contains structured and well-defined information.  
        """

        text = f"""
        Identify and generate precise information for the dataset labels: {self.labels}.  
        Extract **keywords, characteristics, and properties** based on image representation.  
        """

        response = self.query_text(prompt, text)
        response = json.loads(response)
        characteristics = response["Characteristics"]
        return characteristics
    
    def annotator_task(self, image, extra_infos):
        prompt = f"""
            Classify the following text as one of the following types: {self.labels}
            {extra_infos}
            Provide the classification first, followed by a short justification.
            Image: "{image}"
            You will be provided with a text, and you will output a JSON object containing the following information:
            Format:
            Label: <your_predicted_label>
            Explanation: <your_justification>
        """
        response = self.query(prompt, image)
            
        if response is None:
            return "Unknown", "Query response was None"
    
        try:
            response = json.loads(response)
            label = response.get("Label", -1)
            explanation = response.get("Explanation", "No explanation provided")
        except json.JSONDecodeError:
            return "Unknown", "Failed to parse JSON response"
    
        return label, explanation

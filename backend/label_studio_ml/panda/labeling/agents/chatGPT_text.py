import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import json
from openai import OpenAI
import pandas as pd
import json

OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)

class agent:
    def __init__(self, model_name, labels):
        self.model_name = model_name
        self.labels = labels

    def query(self, prompt, content):
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

        return response.choices[0].message.content
    

    
    def find_information(self):
        prompt = f"""
            Your task is to extract and provide structured information for each dataset label.  
            Focus on identifying key **visual** and **textual** attributes that define each label.  

            **Requirements:**  
            - Identify and extract **keywords** related to each label.  
            - Describe the **key characteristics** based on image or text representation.  
            - Specify important **properties**: keywords, content type if text-based).  
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
        Extract **keywords, characteristics, and properties** based on image or text representation.  
        """

        response = self.query(prompt, text)
        response = json.loads(response)
        characteristics = response["Characteristics"]
        return characteristics
    
    def annotator_task(self, text, extra_infos):
        prompt = f"""
            Classify the following text as one of the following types: {self.labels}
            {extra_infos}
            Provide the classification first, followed by a short justification.
            Text: "{text}"
            You will be provided with a text, and you will output a JSON object containing the following information:
            Format:
            Label: <your_predicted_label>
            Explanation: <your_justification>
        """
        response = self.query(prompt, text)
        response = json.loads(response)
        # print(type(response))
        label = response["Label"]
        explantion = response["Explanation"]
        return label, explantion
    
    def challenger_task(self, text, annotation, explanation):
        prompt = f"""
        Text: "{text}"
        The given text has been classified as "{annotation}".
        Explanation provided: {explanation}
        Your task is to challenge this classification. Identify any possible misclassification, ambiguity, or alternative classification with one of the following types: {self.labels}. Only use the provided types.
        You will be provided with a text, and you will output a JSON object containing the following information:
        Format:
        Critique: <your_critique>
        """
        response = self.query(prompt, text)
        response = json.loads(response)
        critique = response["Critique"]
        return critique
    
    def adjudicator_task(self, text, annotation, explanation, critique):
        prompt = f"""
        Text: "{text}"
        Initial annotation: {annotation}
        Explanation: {explanation}
        Challenger's critique: {critique}
        Your task is to make a final decision by weighing both perspectives.
        Provide the final classification for the given text with one of the following types: {self.labels}. Only use the provided types.
        You will be provided with a text, and you will output a JSON object containing the following information:

        Format:
        Final Label: <your_predicted_final_label>
        """
        response = self.query(prompt, text)
        response = json.loads(response)    
        label = response["Final Label"]
        return label

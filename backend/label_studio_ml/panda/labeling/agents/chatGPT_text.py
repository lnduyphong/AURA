import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import json
from openai import OpenAI
import pandas as pd
import json

client = OpenAI()

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
    
    def generate_prompt(self):
        meta_prompt = """
            Generate a structured prompt for an AI model to classify a given text into predefined categories.  
            The prompt should instruct the model to categorize the text based on a list of labels and provide a justification.  
            You will be provided with a text, and you will output a JSON object containing the following information:
            The generated prompt should follow this format:
            Prompt: <your_pormpt>
 
            Example:            
            Prompt:
                Classify the following text as one of the following types: {{self.labels}}
                {{extra_infos}}
                Provide the classification first, followed by a short justification.
                Text: "{{text}}"
                You will be provided with a text, and you will output a JSON object containing the following information:
                Format:
                Label: <your_predicted_label>
                Explanation: <your_justification>
            
            Ensure that the generated prompt is clear, concise, and structured for effective AI response.
        """
        response = self.query(meta_prompt, meta_prompt)
        response = json.loads(response)
        # print(type(response))
        characteristics = response["Prompt"]
        return characteristics

    
    def find_information(self):
        prompt = f"""
            Write extra information describing the labels of the dataset following this structure:
            Characteristics:
                Jasmine rice is long and slender in shape, white with a slightly translucent appearance in color.  
                Karacadag rice is short and slightly oval in shape, white, but more opaque than Jasmine rice.  
                Ipsala rice is medium grain and slightly elongated, white and slightly opaque in color.  
                Arborio rice is short and plump in shape, white and highly opaque in color.  
                Basmati rice is extra-long, slender, and slightly tapered at the ends in shape, white with a pearly luster in color.  
            The labels that need descriptions are {self.labels}.
            You will be provided with a text, and you will output a JSON object containing the following information:
            Format:
            Characteristics: <information describing the labels >
        """
        text = f"""
        You need to find extra information about the labels: {self.labels} in the dataset and describe them in a structured format. 
        Each label should have a clear and structured description, detailing its characteristics, typical topics, and common themes. 
        """
        response = self.query(prompt, text)
        response = json.loads(response)
        # print(type(response))
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
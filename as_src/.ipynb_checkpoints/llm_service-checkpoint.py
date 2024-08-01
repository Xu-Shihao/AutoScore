import base64
import requests
from dotenv import load_dotenv, find_dotenv
import os
import json

def call_gpt4_vision(image_path, prompt):

    _ = load_dotenv(find_dotenv())  # read local .env file
    api_key = os.environ["OPENAI_API_KEY"]

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1,
        # "response_format": {"type": "json_object"}
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        return result.strip()
    else:
        return call_gpt4_vision(image_path, prompt)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def call_claude(image_path, prompt):

    _ = load_dotenv(find_dotenv())  # read local .env file
    api_key = os.environ["ANTHROPIC_API_KEY"]

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    # Encode the image
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 300,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()['content'][0]['text']
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result.strip()
    else:
        return {"error": f"API Error: {response.status_code}", "details": response.text}

def parse_llm_json(text):

    # Remove the ```json and ``` markers if present
    if text.startswith('```json'):
        text = text.split('```json', 1)[1]
    if text.endswith('```'):
        text = text.rsplit('```', 1)[0]
    
    # Strip any leading/trailing whitespace
    text = text.strip()
    
    # Parse the JSON string
    data = json.loads(text)
    
    # Return the data as a dictionary
    return data
    
if __name__ == '__main__':

    img_path = "./OCRAutoScore/example_img/ocr_example1.png"
    print(call_gpt4_vision(img_path, "Describe the answer shown in the image。"))
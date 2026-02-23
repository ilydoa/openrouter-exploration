import os
import pandas as pd
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm


#Collect text posts from selected dataset
DATASET = "../data/input/cleaned_annotations.csv"
df = pd.read_csv(DATASET)
tokens = df["token"].to_list()

load_dotenv()
api_key = os.getenv("API_KEY")
if api_key is None:
    print("Error: API key not found in environment variables.")
else:
    print("API key loaded successfully.")

#Set up client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

#*** QWEN NER ***#
completions = []
for token in tqdm(tokens, desc="Processing"):
    completion = client.chat.completions.create(
        extra_body={},
        model="qwen/qwen3-8b",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"""Identify if the following token is a named entity.
                Reply with:
                "O (no label)" if it is not a named entity
                "B-PER" if it is the beginning part of a person entity
                "I-PER" if it is continuing a person entity
                "B-ORG" if it is the beginning part of an organization entity
                "I-ORG" if it is continuing an organization entity
                "B-MISC" if it is the beginning part of another type of entity
                "I-MISC" if it is continuing another type of entity
            {token}
            """
            },

            ]
        }
        ]
    )
    completions.append(completion.choices[0].message.content)

df["qwen_ner"] = completions

#Save CSV with model outputs
output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'qwen_ner_{timestamp}.csv'
df.to_csv(name)
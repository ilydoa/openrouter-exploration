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


#*** QWEN LID ***#
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
                "text": f"""Identify the language of the following token as Turkish, English, mixed Tur-Eng, or other.
                Reply with either "TR" for Turkish, "EN" for English, "MIXED" for mixed, or "OTHER" for other.
            {token}
            """
            },

            ]
        }
        ]
    )
    completions.append(completion.choices[0].message.content)

df["qwen_langid"] = completions


#Save CSV with model outputs
output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'qwen_lid_{timestamp}.csv'
df.to_csv(name)
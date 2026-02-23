import os
import pandas as pd
import datetime
from openai import OpenAI
#from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

#Collect text posts from selected dataset
DATASET = "../data/input/cleaned_annotations.csv"
df = pd.read_csv(DATASET)
tokens = df["token"].to_list()

#Set up client
client=OpenAI(api_key="api key")

#*** LID: GPT4o ***#
completions = []
for token in tqdm(tokens, desc="Processing LID"):
    prompt = f"""Identify the language of the following token as Turkish, English, mixed Tur-Eng, or other.
                Reply with either "TR" for Turkish, "EN" for English, "MIXED" for mixed, or "OTHER" for other.
            {token}
            """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    completions.append(response.output_text)

df["gpt_langid"] = completions

#*** NER: GPT4o ***#
completions = []
for token in tqdm(tokens, desc="Processing NER"):
    prompt = f"""Identify if the following token is a named entity.
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
    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    completions.append(response.output_text)

df["gpt_ner"] = completions

#Save CSV with model outputs
output_dir = Path("../data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = output_dir / f'gpt_lid_ner_{timestamp}.csv'
df.to_csv(name)
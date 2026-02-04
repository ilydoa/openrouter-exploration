#Necessary imports
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

#Collect text posts from selected dataset
DATASET = "turkish_text_samples.csv"
df = pd.read_csv(DATASET)
text_samples = df["text"].to_list()

#Get API key for prompting
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

#***** QWEN 8B RESPONSE COLLECTION *****#

completions = []
for sample in text_samples:
    completion = client.chat.completions.create(
        extra_body={},
        model="qwen/qwen3-8b",
        messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f""" You will be given a Turkish social media post.
            Write a response post to it.
            
            Here is the text post to respond to:
            {sample}
            """
            },

            ]
        }
        ]
    )
    completions.append(completion.choices[0].message.content)

df["qwen_responses"] = completions

#***** LLAMA 8B INSTRUCT RESPONSE COLLECTION *****#

completions = []

for sample in text_samples:
  completion = client.chat.completions.create(
    extra_body={},
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[
      {
        "role": "user",
        "content":f""" You will be given a Turkish social media post.
          Write a response post to it. Do not include any text other than your response post.
          
          Here is the text post to respond to:
          {sample}
          """
      }
    ]
  )

  completions.append(completion.choices[0].message.content)

df["llama_responses"] = completions

#Save CSV with model outputs
df.to_csv("post_responses_qwen_llama.csv")


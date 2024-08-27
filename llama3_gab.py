from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('ghc_test.tsv', sep='\t')
df = df.head(5)

responses = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for index, row in df.iterrows():
    text = row['text']
    messages = [
        {"role": "system", "content": "You are an annotator for hate speech detection."},
        {"role": "user", "content": """Read carefully the definition of 'hate speech' provided. 
Your task is to classify the input text as containing hate speech or not. You can only rely on the definition provided.
Respond only with YES or NO.

DEFINITION:
Language that intends to — through rhetorical devices and contextual references — attack the dignity of a group of people, either through an incitement to violence, encouragement of the incitement to violence, or the incitement to hatred.

TEXT: {text}

ANSWER:
""".format(text=text)}]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0,
)

    response = outputs[0]["generated_text"][len(prompt):]
    responses.append(response)

df['response'] = responses
df.to_csv('llama3_gab_trial.csv', index=False)


#!/usr/bin/env python
# coding: utf-8
import sklearn
import torch
import re
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
from torch import cuda
from tqdm import tqdm
from utils.exec_sql import select_sql

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "./models/deepseek-coder-6.7b-instruct-v1.5"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # use with amper architecture
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config, # use when low on memory
    device_map="auto"
)

model = PeftModel.from_pretrained(model, "./result/stage_2/final_checkpoint",
                                  torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.encode(' ;')

from transformers import StoppingCriteria


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[6203]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def append_string_to_file(text, file_path):
    with open(file_path, 'a') as file:
        file.write(text + '\n')


def remove_spaces(text):
    return re.sub(r'\s+', ' ', text)


def call_mistral(inputs):
    output_tokens = model.generate(inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                   eos_token_id=tokenizer.eos_token_id, stopping_criteria=[EosListStoppingCriteria()])
    return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

df = pd.read_csv("./data/spider_dev_dataset.csv")
results = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    query = row['query']
    database_schema = row['database_schema_ft']
    db_id = row['db_id']
    user_message = f"""You are a text-to-SQL expert. Given the following SQL tables, your job is to generate the Sqlite SQL query given the user's question. Put your answer inside the ```sql and ``` tags.
{database_schema}
###
Question: {question}
"""
    messages = [
        {"role": "user", "content": user_message.strip()}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=5,
        num_return_sequences=5
    )
    response_list = []
    for i, generated in enumerate(generated_ids):
        generated = [generated]
        generated = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated)]
        response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        if "```sql" in response:
            response = response.split("```sql")[1]
            if "```" in response:
                response = response.split("```")[0]
        response = response.split(" ;")[0]
        response = re.sub(r'\s+', ' ', response).strip()
        response_list.append(response)
    db_path = './data/database/'+db_id+'/'+db_id+'.sqlite'
    result = select_sql(response_list, db_path)
    print('\n')
    print(result)
    print(query)
    print("============================")
    results.append([result, query, row['question'], row['db_id']])

new_df = pd.DataFrame(results, columns=['generated_query', 'reference_query', 'question', 'db_id'])
with open('./output/result.txt', 'w') as file:
    for index, row in new_df.iterrows():
        file.write(remove_spaces(row['generated_query']) + '\n')
file.close()


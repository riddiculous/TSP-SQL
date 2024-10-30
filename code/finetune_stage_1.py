#!/usr/bin/env python
# coding: utf-8
import sklearn
import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType
from datasets import load_dataset
from sql_metadata import Parser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm


def formatting_prompts_func(training_dataset):
    output_texts = []
    for i in range(len(training_dataset['question'])):
        question = training_dataset['question'][i]
        correct_tables = training_dataset['correct_tables'][i]
        correct_tables = " , ".join(set(correct_tables.split(", ")))

        correct_columns = training_dataset['correct_columns'][i]
        if "none" in correct_columns:
            correct_columns = "*"
        else:
            correct_columns = " , ".join(set(correct_columns.split(", ")))

        correct_values = training_dataset['correct_values'][i]
        if correct_values is None:
            correct_values = "none"
        else:
            correct_values = " , ".join(set(correct_values.split(", ")))

        correct_clause_keywords = training_dataset['correct_clause_keywords'][i]
        correct_clause_keywords = " , ".join(set(sorted(correct_clause_keywords.split(", "), key=keywords_.index)))

        correct_where_operands = training_dataset['correct_where_operands'][i]
        correct_where_operands = " , ".join(set(correct_where_operands.split(", ")))

        correct_aggregate_functions = training_dataset['correct_aggregate_functions'][i]
        correct_aggregate_functions = " , ".join(set(correct_aggregate_functions.split(", ")))

        correct_nested_queries = training_dataset['correct_nested_queries'][i]

        database_schema = training_dataset['database_schema_ft'][i]

        user_message = f"""Given the following SQL tables and a question:
{database_schema}
###
Question: {question}
You are a text-to-SQL expert. Before translating the question to SQL, finish the following tasks based on the given SQL tables and question:
1. Determine the tables and columns that the question is referring to.
2. Decide the SQL value if needed.
3. Choose the SQL clause keywords.
4. Choose the operands in "where" clause of SQL.
5. Choose the aggregate functions of SQL.
6. Decide whether the SQL needs nested queries or not.
"""
        assitant_message = f"""tables: {correct_tables}
columns: {correct_columns}
values: {correct_values}
clause keywords: {correct_clause_keywords}
where operands: {correct_where_operands}
aggregate functions: {correct_aggregate_functions}
nested queries: {str(correct_nested_queries).lower()}
"""
        messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assitant_message},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        output_texts.append(text)
    return output_texts

keywords_ = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY(asc)', 'ORDER BY(desc)', 'LIMIT(1)', 'LIMIT(2)','LIMIT(3)','LIMIT(4)','LIMIT(5)','LIMIT(6)','LIMIT(7)','LIMIT(8)','LIMIT(9)','LIMIT(10)','INTERSECT', 'UNION', 'EXCEPT']
keywords_ = [word.lower() for word in keywords_]

model_name = "./models/deepseek-coder-6.7b-instruct-v1.5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype = torch.bfloat16,
    device_map="auto"
)

model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

print(model)

data_files = {"train": "./data/spider_train_dataset.csv", "validation": "./data/spider_dev_dataset.csv"}
dataset = load_dataset('csv', data_files=data_files)
response_template = "### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

lora_r = 8
lora_alpha = 16
lora_dropout = 0.1
output_dir = "./result/stage_1/"
num_train_epochs = 3
bf16 = True
overwrite_output_dir = True
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 16
gradient_checkpointing = True
# evaluation_strategy = "steps"
learning_rate = 5e-5
weight_decay = 0.01
lr_scheduler_type = "cosine"
warmup_ratio = 0.01
max_grad_norm = 0.3
group_by_length = True
auto_find_batch_size = False
# save_steps = 50
evaluation_strategy = "epoch"
save_strategy = "epoch"
logging_steps = 50
load_best_model_at_end= False
packing = False
save_total_limit=3
neftune_noise_alpha=5
report_to="tensorboard"
max_seq_length = 2500

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"
    ],
    task_type=TaskType.CAUSAL_LM,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs=num_train_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
    evaluation_strategy=evaluation_strategy,
    max_grad_norm = max_grad_norm,
    auto_find_batch_size = auto_find_batch_size,
    save_total_limit = save_total_limit,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy = save_strategy,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to,
    neftune_noise_alpha=neftune_noise_alpha
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'].shuffle(seed=42),
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=max_seq_length,
    packing=packing
)

trainer.train()

output_dir = os.path.join("./result/stage_1/", "final_checkpoint")
trainer.model.save_pretrained(output_dir)


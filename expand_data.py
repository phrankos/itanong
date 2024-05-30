import json
import csv
import os

# Step 1: Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch
from transformers import BitsAndBytesConfig
from datasets import concatenate_datasets
from multiprocessing import cpu_count
import random
import re
from transformers import AutoTokenizer
from datasets import DatasetDict
from datasets import load_dataset
import os
import time
import json


def data_collator(products):
    instruction = """
You are a helpful assistant that generates paraphrased questions and answers based on given product information. You will be given a question-and-answer prompt which will have the delimiters <info_start> and <info_end>. Your task is to generate paraphrased questions and answers based on the question-answer pair.
    """

    combined_scraped_data = []

    for i in products:
        prompt = {}
        prompt['instruction'] = instruction.strip()
        prompt['input'] = """ 
Hello, I would like to ask you to generate a dataset about tariff rates or duty rates of a product. I will provide you with the I will provide you with the product description, and your task is to generate a dictionary with the following format:
```
{

"question": "If I export 1000kg of fresh mangoes from Guimaras to Manila, how much dutiable value/freight value/customs duty should I pay?",
"response":
{
    "product_description": "Mangoes, fresh",
    "task": "duty_rate"
}

Please generate a minimum of 100 datapoints using the given the context. I will not be able to use the data if it is not in the specific format given. Please follow the given format strictly and do not respond with anything other than the requested answer. Avoid using any special characters in your answers.
Use the delimeters "<data_start>" and "<data_end>" to separate the generated data. Please follow the format strictly.

I have an example of the format that you will follow. Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:

<example>
User: 
<info_start>
Product: Mangoes, fresh
<info_end>

Assistant: 
<data_start>
[
{
    "question": "If I export 1000kg of fresh mangoes from Guimaras to Manila, how much dutiable value/freight value/customs duty should I pay?",
    "response":
    {
        "product_description": "Mangoes, fresh",
        "task": "duty_rate"
    }
},
{
    "question": "How much is the customs duty for fresh mangoes?",
    "response":
    {
        "product_description": "Mangoes, fresh",
        "task": "duty_rate"
    }
}
]
<data_end>

</example>
"""
        new_VAL = f"""
Product: {i}
        """
        prompt['context'] = f"<info_start>{new_VAL}<info_end>"
        combined_scraped_data.append(prompt)

    return combined_scraped_data


def ready_chat(example):
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": "Ok. I understand. After this message, I will not reply with anything other than the generated dataset."},
        {"role": "user", "content": example["context"]}
    ]

    return messages


def get_categories():
    categories = []
    with open('Categories.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            categories.append(row)
    return categories

def get_products():
    products = []
    with open('Products.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            products.append(row[4])
    return products

def main(index):
    # Log in  to Hugging Face
    os.system('huggingface-cli login --token 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')
    os.system('huggingface-cli whoami')

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Load the model
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device="cpu",

        # quantization_config=bnb_config  # un-comment to quantize your model
    )

    tokenizer.pad_token = tokenizer.eos_token
    categories = get_categories()
    products = get_products()[1:]


    # Collate the data
    combined_scraped_data = data_collator(products)

    # Generate the dataset
    generated_dataset = []
    counter = 0
    for prompt in combined_scraped_data:
        t = time.time()

        print(
            f"Generating sample {counter + 1} of {len(combined_scraped_data)}. Length of prompt: {len(prompt['context'])}")
        input_text = ready_chat(prompt)

        inputs = tokenizer.apply_chat_template(
            input_text, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            inputs,
            max_new_tokens=5000,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            do_sample=True,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_text = generated_text.replace(
            tokenizer.decode(inputs[0], skip_special_tokens=True), "")
        # print(generated_text)

        generated_dataset.append(generated_text)

        # Split the generated text into individual samples
        counter += 1
        print(f"Time taken: {time.time() - t}")

        # Save the generated dataset to a JSON file
        save_path = f"llama3-gendata{index}.json"
    with open(save_path, "w") as f:
        json.dump(generated_dataset, f, indent=4)


if __name__ == "__main__":
    
    for i in range(0, 10):
        start = time.time()
        main(i)
        print(
            f"Generated data llama3-gendata{i}.json successfully, took {time.time() - start} seconds.")
        # clear the cache
        torch.cuda.empty_cache()
        print("Cleared cache.")
        time.sleep(5)

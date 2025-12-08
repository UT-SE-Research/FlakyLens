import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import sys
from utils import set_seed, setup_logging, seed_worker, train, evaluate, parse_cr, qwen_model_define, parse_category_and_token_list, init_setup, contains_english_letter, filter_tokens, forward_func, predict, deep_seek_coder_model_define, llama3_8b_model_define, codegemma7b_model_define, gemma2b_model_define, gemma7b_model_define, interpret_with_ig_qwen, interpret_with_ig_gemma7b, codellama_7b_instruct_model_define, interpret_with_ig_codellama, make_few_show_example
from data_processing import collect_token_attention_scores, read_data, converting_df_to_series
import pandas as pd
import os
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import re
from collections import Counter
import torch.nn.functional as F
#from Testing_gemma_7b_categorization import parse_generated_output_to_get_category
#from Testing_gemma_2b_categorization  import identify_test_category
import transformers
import torch

token = os.getenv("HF_TOKEN")
if token is None:
    raise ValueError("Environment variable HF_TOKEN not set. Please export it before running.(export HF_TOKEN=hf_your_token_here)")

login(token=token)

timeStart = time.time()

import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import numpy as np
def categories_defination_and_tokenizers_max_length():
    categories = {
    "async wait": 0,
    "concurrency": 1,
    "time": 2,
    "unordered collection": 3,
    "order dependent test": 4,
    "not flaky": 5
    }

    MAX_LENGTH = 2048  # Or 8192, based on your model
    return categories, MAX_LENGTH 

def prompt_definition():
    definitions = f"""You are an expert at identifying flaky tests and analyzing their type. Flaky tests are tests that pass and fail non-deterministically for the same code. You are given a test, and you have to check whether it is flaky or not. If it is flaky, you have to identify the type of flakiness and classify into one of the following categories or just say Not Flaky: 
    1. Async wait: The test execution makes an asynchronous call and does not properly wait for the result of the call to be available before proceeding. This can lead to non-deterministic test outcomes. 
    2. Concurrency: Test non-determinism is due to different threads interacting in a non-desirable manner (but not due to asynchronous calls from the Async Wait category), e.g., due to data races, atomicity violations, or deadlocks. 
    3. Time: Relying on the system time introduces non-deterministic failures, e.g., a test may fail when the midnight changes in the UTC time zone. Some tests also fail due to the precision by which time is reported as it can vary from one platform to another. 
    4. Unordered collection: In general, when iterating over unordered collections (e.g., sets), the code should not assume that the elements are returned in a particular order. If it does assume, the test outcome can become non-deterministic as different executions may have a different order.
    5. Test Order dependent test: The test depends on the order of execution of other tests. If the order changes, the test outcome may change. 
    6. Not Flaky: The test is not flaky due to the above reasons."""
    return definitions

def parse_generated_output_to_get_category(output_text):
    # Find all occurrences of "Category:"
    matches = re.findall(r"Category:\s*([^\n]*)", output_text)
    
    if matches:
        output_category = matches[-1].strip()  # Take the last occurrence
        output_category = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", output_category)  # Remove non-alphanumeric chars at start/end
        #output_category = re.sub(r"^\s+", "", output_category)  # Explicitly remove leading spaces
        #output_category = output_category.replace("\xa0", "").replace("\u200b", "")


        #output_category = output_category.lstrip()  # Take the last occurrence
    else:
        output_category = "Unknown"

    print(len(output_category))  # Should be the exact length of "Concurrency" (11)
    print(repr(output_category))  # To see hidden characters
    print(f">>>{output_category}<<<")  # This will show if there are hidden spaces

    print("*****Extracted Category:",output_category)
    
    return output_category 

def give_test_data_in_chunks_codellama(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique): #BERT
    x_test_df = pd.DataFrame(x_test_nparray)
    n = 1 #len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    model.eval()
    total_attributions = []
    total_tokens = []
    total_preds = []
    vis_data_records_ig = []
    category_token_map = {}  # Dictionary to store tokens per category
    count = 0

    top_tokens_per_test = []
    category_token_map = {}  # Dictionary to store tokens per categoy
    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()
    MAX_LENGTH = 1024
    examples=make_few_show_example(fold) 
    for index, row in x_test_df.iterrows():
        test_data = row['full_code']
        #prompt = "What is the category of the given test? Is it Async wait or Concurrency or Time or Unordered collection or Order dependent test or non-flaky? \n" 
        #input_str = prompt + test_data 
        #category_list = ["Async-wait", "Concurrency", "Time", "Unordered-collection", "Order-dependent", "non-flaky"]

#        prompt = f"""
#Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or non-flaky.
#
#Look at the behavior shown in the test method and decide the most appropriate category it belongs to.
#
#Here are labeled examples:
#
#{examples}
#
#Now classify the following test:
#
#Test:
#{test_data}
#
#**Output Format (MUST follow this format exactly):**
#```
#**Category**: <one of the six categories above>
#``` 
#        """
        prompt = f"""
You are an expert of flaky test classification. Your task is to classify the given Java unit test into exactly one of the following categories:

The possible categories are:
- Async wait
- Concurrency
- Time
- Unordered collection
- Order dependent test
- Not Flaky

Here are some labeled examples:

{examples}

Now classify the following test:

Test:
{test_data}

Your output must be in the following format, with no extra text:

**Category**: <one of the six categories above>
"""

        #print(prompt)
        # Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
        

        model_inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=MAX_LENGTH, truncation=True).to(device)
        # Move inputs to the correct device
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}  # Ensure it's a dict

        # Limit tokens to avoid excessive generation
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=15,  # Limit to prevent explanations
            do_sample=False, 
            temperature=0.0,  # Ensures deterministic behavior
            eos_token_id=tokenizer.eos_token_id,  # Ensure proper stopping
            pad_token_id=tokenizer.pad_token_id,  
            return_dict_in_generate=True,  # Return full outputs
            output_scores=True  
        )    

        input_length = model_inputs["input_ids"].shape[1]
        raw_output = tokenizer.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True)[0]
        #print(raw_output)

        # Extract only the category using regex
        match = re.search(r'\*\*Category\*\*:\s*(.*)', raw_output)
        category = match.group(1).strip() if match else "Unknown"

        print(f"Predicted Category: {category}") 
        #outputs = model.generate(
        #    **model_inputs,
        #    max_new_tokens=1000, 
        #    do_sample=False, 
        #    top_k=50, 
        #    top_p=0.95, 
        #    temperature=0.8,
        #    num_return_sequences=1, 
        #    pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id explicitly
        #    eos_token_id=tokenizer.eos_token_id,
        #    output_attentions=True  # Capture attention weights
        #)

        #input_length = model_inputs["input_ids"].shape[1]
        #output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

        #print(output)
        #output_categories = output.split("\n")
        #print(output_categories)

        #category, tokens = parse_category_and_tokens(output[0])
        #output_category = output_categories[0]
        #print('Category_name=',category)
        #print('token=', tokens)

        #category_value = categories.get(category.lower().strip(), 6)  # Return -1 if category not found
        output_category_lower = category.lower()
        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)

        # Print only the tokens (no scores)
        #print("\n✅ Top-20 Tokens Based on Attribution Scores:")
        #print(top_token_list)

        category_value = categories.get(output_category_lower, 6)  # Return -1 if category not found
        print('category_value=')
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

        #print("\nFinal Category-Token Map:", category_token_map)
        #exit()
    #return total_preds, category_token_map
    return total_preds, category_token_map, top_tokens_per_test
    #print(category_value)
    

    #return total_preds, category_token_map

def give_test_data_in_chunks_deep_seek_coder(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique): 
    x_test_df = pd.DataFrame(x_test_nparray)
    y_test_df = pd.DataFrame(y_test_nparray, columns=['category'])  # Convert to DataFrame
    #print(y_test_df)
    n = 1  # len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    total_preds = []
    top_tokens_per_test = []
    category_token_map = {}  # Dictionary to store tokens per categoy
    count = 0
    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()

    MAX_LENGTH = 1600  # Or 8192, based on your model
    examples=make_few_show_example(fold) 
    model.eval()  # Set the model to evaluation mode
    #for index, row in x_test_df.iterrows():
    for index, (test_data, actual_label) in enumerate(zip(x_test_df['full_code'], y_test_df['category'])):
        print("****fold=", fold)
        definitions = prompt_definition()

        prompt = f"""
Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or not flaky.

Here are some leveled examples:
{examples}

Now classify the following test:
Test:
{test_data}                        

**Output Format (MUST follow this format exactly):**
```
**Category**: <one of the six categories above>
``` 
"""

        messages = [
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
        outputs = model.generate(
            #inputs, 
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512, 
            do_sample=False, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.8,
            num_return_sequences=1, 
            pad_token_id=tokenizer.pad_token_id,  # Set pad_token_id explicitly
            eos_token_id=tokenizer.eos_token_id
        )
        #output_category = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        # Decode and clean the output
        raw_output = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True).strip()
        #print("Raw model output:", raw_output)

        # Extract only the category using regex
        match = re.search(r'\*\*Category\*\*:\s*(.*)', raw_output)
        category = match.group(1).strip() if match else "Unknown"

        print(f"Predicted Category: {category}") 

        #print("CATEG=",category)
        # Check if "sequences" exists in outputs
        '''if isinstance(outputs, torch.Tensor):
            print("outputs is a torch.tensor")
            sequences = outputs  # If `outputs` is a tensor, use it directly
            # Extract generated tokens from outputs (excluding input tokens)
            generated_tokens = outputs[:, input_ids.shape[1]:]  # Slice out the new tokens
            
            # Decode the generated tokens
            output_text = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True).strip()
            
            print("Decoded Output:", output_text)

        print("Raw model output:", output_text)

        category = parse_category_and_token_list(output_text)'''
        #print("category=", category)

        output_category_lower = category.lower()
        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)

        category_value = categories.get(output_category_lower, 6)  # Return -1 if category not found
        print('category_value=')
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

    return total_preds, category_token_map, top_tokens_per_test

def give_test_data_in_chunks_llama3_8b(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique):
    x_test_df = pd.DataFrame(x_test_nparray)
    n = 1 #len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    model.eval()
    total_attributions = []
    total_tokens = []
    top_tokens_per_test = []
    total_preds = []
    vis_data_records_ig = []
    category_token_map = {}  # Dictionary to store tokens per category
    count = 0
    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()
    MAX_LENGTH = 1600
    y_test_df = pd.DataFrame(y_test_nparray, columns=['category'])  # Convert to DataFrame
    #for index, row in x_test_df.iterrows():
    for index, (test_data, actual_label) in enumerate(zip(x_test_df['full_code'], y_test_df['category'])):
        prompt = f"""
        Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or not flaky.

        Think step by step to determine the most appropriate category.

        Test:
        {test_data}
         **Output Format (MUST follow this format exactly):**
        ```
        Category: <one of the six categories above>
        ``` 
        """

        definitions = prompt_definition() 

        messages = [
        {"role": "system", "content": definitions}, #I can specify the output format as the json
        {"role": "user", "content": prompt},
        ]
       
        #model_inputs = tokenizer.apply_chat_template(
        #    messages,
        #    return_tensors="pt",
        #    max_length=MAX_LENGTH,  # Truncate long inputs
        #        truncation=True
        #).to(model.device)

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)


        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        if isinstance(model_inputs, transformers.tokenization_utils_base.BatchEncoding):
            model_inputs = dict(model_inputs)  # Explicitly convert to dictionary
        
        model_inputs = {key: value.to(model.device) for key, value in model_inputs.items()}

        outputs = model.generate(
            **model_inputs,
            #attention_mask=attention_mask,
            max_new_tokens=50,
            #max_length=25,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.8,
            top_p=1,
            return_dict_in_generate=True,  # Return full output including attention
            output_attentions=True  # Capture attention weights
        )
        #response = outputs[0][model_inputs.shape[-1]:]
        response = outputs["sequences"][0][model_inputs["input_ids"].shape[-1]:]

        output_category = tokenizer.decode(response, skip_special_tokens=True)
        print('output_category=',output_category)
        category = parse_category_and_token_list(output_category)
        print("Extracted Category:", category)
        output_category_lower = category.lower()
        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)
        category_value = categories.get(output_category_lower, 6)  # Return -1 if category not found
        print('category_value=')
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

    return total_preds, category_token_map, top_tokens_per_test


def give_test_data_in_chunks_codegemma(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique): #BERT
    #max_length = 1024; 128
    max_length = 1024
    x_test_df = pd.DataFrame(x_test_nparray)
    n = 1 #len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    model.eval()
    total_attributions = []
    total_tokens = []
    total_preds = []
    vis_data_records_ig = []
    count = 0

    top_tokens_per_test = []
    category_token_map = {}  # Dictionary to store tokens per category

    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()

    for index, row in x_test_df.iterrows():
        test_data = row['full_code']

        prompt = f"""
        Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or not flaky.

        Think step by step to determine the most appropriate category.

        Test:
        {test_data}
         **Output Format (MUST follow this format exactly):**
        ```
        Category: <one of the six categories above>
        ``` 
        """
        definitions = prompt_definition 

        text = [
        #{"role": "system", "content": definitions}, #I can specify the output format as the json
        {"role": "user", "content": prompt},
        ]

        model_inputs = tokenizer.apply_chat_template(text, return_tensors="pt",
            max_length=MAX_LENGTH,  # Truncate long inputs
                truncation=True
        ).to(model.device)
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        
        outputs = model.generate(
            input_ids=model_inputs,
            #attention_mask=attention_mask,
            max_new_tokens=50,
            #max_length=25,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.8,
            return_dict_in_generate=True,  # Return full output including attention
            output_attentions= False #True  # Capture attention weights
        )

        #response = outputs[0][model_inputs.shape[-1]:]
        response = outputs["sequences"][:, model_inputs.shape[-1]:]

        #print(response)
        output_text = tokenizer.decode(response[0], skip_special_tokens=True)
        #print(output_text)
        output_category = parse_generated_output_to_get_category(output_text)
        output_category_lower = output_category.lower()

        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)

        # Print only the tokens (no scores)
        #print("\nTop-20 Tokens Based on Attribution Scores:")
        #print(top_token_list)
        #Storing tokens for each category test 
        #category = parse_category_and_token_list(output_category)

        category_value = categories.get(output_category_lower, 6)  # Return -1 if category not found
        print('category_value=')
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

        #print("\nFinal Category-Token Map:", category_token_map)

    #return total_preds
    return total_preds, category_token_map, top_tokens_per_test

    

def give_test_data_in_chunks_gemma2b(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique): #BERT
    print('FROM GEMMA')
    max_length = 512
    x_test_df = pd.DataFrame(x_test_nparray)
    n = 1 #len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    model.eval()
    total_tokens = []
    total_preds = []
    count = 0
    top_tokens_per_test = []
    category_token_map = {}  # Dictionary to store tokens per category

    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()

    for index, row in x_test_df.iterrows():
        test_data = row['full_code']
        flag = False
        #while True:
        #output_category_lower = identify_test_category(test_data, tokenizer, model, device)

        category_list=["Async-wait", "Concurrency", "Time", "Unordered-collection", "Order-dependent", "not flaky"]
        prompt=f"""Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or not flaky.        

        Think step by step to determine the most appropriate category.

        Test:
        {test_data}
         **Output Format (MUST follow this format exactly):**
        ```
        Category: <one of the six categories above>
        ``` 
        """

        model_inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.3,
            return_dict_in_generate=True,
            output_attentions=True
        )

        output_text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
        #print('output=', output_text)
        #output = output.split("Category:", 1)[1]
        output_category = parse_generated_output_to_get_category(output_text)
        print(output_category)
        #print(output)
        output_category_lower = output_category.lower()
        print(output_category_lower) 

        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)

        # Print only the tokens (no scores)
        #print("\n Top-20 Tokens Based on Attribution Scores:")
        #print(top_token_list)
        #Storing tokens for each category test 
        #category = parse_category_and_token_list(output_category)

        category_value = categories.get(output_category_lower, 6)  # Return -1 if category not found
        print('category_value=')
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

        #print("\nFinal Category-Token Map:", category_token_map)

    #return total_preds
    return total_preds, category_token_map, top_tokens_per_test


def give_test_data_in_chunks_gemma7b(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique): #BERT
    print('FROM GEMMA')
    x_test_df = pd.DataFrame(x_test_nparray)
    n = 1 #len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    model.eval()
    total_attributions = []
    total_tokens = []
    total_preds = []
    vis_data_records_ig = []
    top_tokens_per_test = []
    category_token_map = {}  # Dictionary to store tokens per category
    count = 0
    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()
    y_test_df = pd.DataFrame(y_test_nparray, columns=['category'])  # Convert to DataFrame
    for index, (test_data, actual_label) in enumerate(zip(x_test_df['full_code'], y_test_df['category'])):
    #for index, row in x_test_df.iterrows():
        #test_data = row['full_code']

        template=f"""You are an expert at identifying flaky tests and analyzing their type. Flaky tests are tests that pass and fail non-deterministically for the same code. You are given a test, and you have to check whether it is flaky or not. If it is flaky, you have to identify the type of flakiness and classify into one of the following categories or just say Not Flaky: 1. Async wait: The test execution makes an asynchronous call and does not properly wait for the result of the call to be available before proceeding. This can lead to non-deterministic test outcomes. 2. Concurrency: Test non-determinism is due to different threads interacting in a non-desirable manner (but not due to asynchronous calls from the Async Wait category), e.g., due to data races, atomicity violations, or deadlocks. 3. Time: Relying on the system time introduces non-deterministic failures, e.g., a test may fail when the midnight changes in the UTC time zone. Some tests also fail due to the precision by which time is reported as it can vary from one platform to another. 4. Unordered collection: In general, when iterating over unordered collections (e.g., sets), the code should not assume that the elements are returned in a particular order. If it does assume, the test outcome can become non-deterministic as different executions may have a different order. 5. Test Order dependent test: The test depends on the order of execution of other tests. If the order changes, the test outcome may change. 6. Not Flaky: The test is not flaky due to the above reasons.

        Test:
        {test_data}

        The output should be only the category name
        CATEGORY:"""

        definitions = prompt_definition()

        prompt = f"""
        Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or not flaky.

        Think step by step to determine the most appropriate category.

        Test:
        {test_data}                        
        **Output Format (MUST follow this format exactly):**
        ```
        Category: <one of the six categories above>
        ``` 
        """
        model_inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(model.device)

        outputs = model.generate(
            **model_inputs,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.3,
            return_dict_in_generate=True,
            output_attentions=False
        ) 

        output_text = tokenizer.decode(outputs["sequences"][0], skip_special_tokens=True)
        #print(output_text)
        output_category = parse_generated_output_to_get_category(output_text)
        #print("category_output=", output_category)



        #NEW PART
        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)
        #exit()

        # Print only the tokens (no scores)
        #print("\n✅ Top-20 Tokens Based on Attribution Scores:")
        #print(top_token_list)
        #Storing tokens for each category test 
        #category = parse_category_and_token_list(output_category)

        category_value = categories.get(output_category.lower().strip(), 6)  # Return -1 if category not found
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

        #print("\nFinal Category-Token Map:", category_token_map)
        #exit()

    return total_preds, category_token_map, top_tokens_per_test

def collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique):
    if ml_technique == "codegemma":
        # Ensure model_inputs is a dictionary
        if isinstance(model_inputs, torch.Tensor):
            model_inputs = {"input_ids": model_inputs}

        # ✅ Add 'attention_mask' if missing
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])  # Create a mask of 1s

    outputs = model(**model_inputs, return_dict=True, output_attentions=True)

    #print(outputs.keys())  # Should contain 'logits'
    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    #print(logits)
    predicted_tokens = logits.argmax(dim=-1)  # Get the token IDs
    #print('predicted_tokens=', predicted_tokens)
    probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
    #print('probs=',probs)
    #print('model=', model)
    if ml_technique == "gemma7b" or ml_technique == "gemma2b" or ml_technique == "codegemma":
        sorted_token_attributions = interpret_with_ig_gemma7b(prompt, tokenizer, model, model_inputs, predicted_tokens, test_data)
    elif ml_technique == "codellama":
        sorted_token_attributions = interpret_with_ig_codellama(prompt, tokenizer, model, model_inputs, predicted_tokens, test_data)

    elif ml_technique == "qwen" or ml_technique == "llama3_8b" or ml_technique == "deep_seek_coder":
        sorted_token_attributions = interpret_with_ig_qwen(prompt, tokenizer, model, model_inputs, predicted_tokens, test_data, ml_technique)
        #print("IIIIII AMMM ********")

    top_token_list = filter_tokens(sorted_token_attributions)
    return top_token_list

def give_test_data_in_chunks_qwen(x_test_nparray, tokenizer, model, batch_size, device, fold, y_test_nparray, ml_technique): 
    # BERT-like max_length
    max_length = 1024
    x_test_df = pd.DataFrame(x_test_nparray)
    n = 1  # len(x_test) / batch_size
    preds_chunks = None
    paired_data = []
    total_preds = []
    category_token_map = {}  # Dictionary to store tokens per category
    count = 0
    top_tokens_per_test = []
    categories, MAX_LENGTH = categories_defination_and_tokenizers_max_length()
    examples=make_few_show_example(fold) 
    model.eval()  # Set the model to evaluation mode
    for index, row in x_test_df.iterrows():
        test_data = row['full_code']

        # Define the prompt and the input for classification
        prompt = f"""
Classify the given test as one of the following categories: Async wait or Concurrency or Time or Unordered collection or Order dependent test or not flaky.

Below are some examples:

{examples}

Now classify the following test:
Test:
{test_data}                        
**Output Format (MUST follow this format exactly):**
```
Category: <one of the six categories above>
```
"""
        definitions = prompt_definition()
        messages = [
            {"role": "system", "content": definitions},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
               messages,
               tokenize=False,
               add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt",
            max_length=MAX_LENGTH,  # Truncate long inputs
                truncation=True
        ).to(model.device)
        #with torch.no_grad():
        outputs = model.generate(
                **model_inputs,
                max_new_tokens=10,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Use deterministic sampling
                temperature=0.8,  # Slightly lower temperature for more focused predictions
                return_dict_in_generate=True,  # Return full output including attention
                output_attentions = False #True  # Capture attention weights
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)
        ]
        output_category = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(output_category)
        #Calculating IG
        top_token_list = "" #collect_token_list_by_applying_ig(model_inputs, prompt, tokenizer, model, test_data, ml_technique)

        category = parse_category_and_token_list(output_category)
        category_value = categories.get(category.lower().strip(), 6)  # Return -1 if category not found
        print(category_value)
        total_preds.append(category_value)
        top_tokens_per_test.append(top_token_list)

        if category_value not in category_token_map:
            category_token_map[category_value] = []  # Initialize empty list if not exists

        if top_token_list:
            category_token_map[category_value].extend(top_token_list)  # Append tokens for the category
        else:
            category_token_map[category_value] = []  # Store empty list for large test cases

        #print("\nFinal Category-Token Map:", category_token_map)
        #exit()

    return total_preds, category_token_map, top_tokens_per_test


def run_experiment(dataset_path, model_weights_path, results_file, data_name_dir, technique):
    device, ml_technique, dataset_category, output_layer, where_data_comes = init_setup(technique, data_name_dir)
    
    if ml_technique == "qwen":
        #print('I am qwen')
        model_name, tokenizer, auto_model = qwen_model_define()
    elif ml_technique == "gemma7b":
        model_name, tokenizer, auto_model = gemma7b_model_define() 
    elif ml_technique == "gemma2b":
        model_name, tokenizer, auto_model = gemma2b_model_define()
    elif ml_technique == "codegemma":
        model_name, tokenizer, auto_model = codegemma7b_model_define()
    elif ml_technique == "llama3_8b":
        model_name, tokenizer, auto_model = llama3_8b_model_define()
    elif ml_technique == "deep_seek_coder":
        model_name, tokenizer, auto_model = deep_seek_coder_model_define()
    elif ml_technique == "codellama":
        model_name, tokenizer, auto_model =  codellama_7b_instruct_model_define()
    else:
        print('model name not correct')
        #exit()
    execution_time = time.time()
    print("Start time of the experiment", execution_time)
    #no_splits = 10 # For FlakiCat=4, IDOFT=10
    TN = FP = FN = TP = 0
    project_group = 0
    total_execution_time = 0
    #no_split = 5
    global_category_token_map = {}
    predictions_per_project_group = {}
    ground_truth_per_project_group = {}
    tokens_per_project_group = {}
    Org_test_per_project_group = {}
    #print(len(input_data))

    # Get all train-test file pairs from the directory
    train_files = sorted([f for f in os.listdir(data_name_dir) if f.startswith("train_") and f.endswith(".csv")])
    test_files = sorted([f for f in os.listdir(data_name_dir) if f.startswith("test_") and f.endswith(".csv")])

    for train_file, test_file in zip(train_files, test_files):
        project_group +=1
        if project_group == 5:
            break

        if os.path.exists("FlakyLens_Categorization-result/score_project_group"+str(project_group)+"_Class.txt"):
            os.remove("FlakyLens_Categorization-result/score_project_group"+str(project_group)+"_Class.txt")
        fit_time=0
        bert_flag=0
        total_execution_time = 0
        feature_extraction_time=0
        #total_execution_time_for_feature_extraction = 0
        print(" NOW IN FOLD NUMBER", project_group)
    
        '''X_train_df = pd.read_csv(data_name+'/data_splits/X_train_project_group'+str(project_group)+'.csv')
        Y_train_df = pd.read_csv(data_name+'/data_splits/y_train_project_group'+str(project_group)+'.csv')
    
        #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'deadcode_perturbation_Most_important_features.csv')
        #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'printStatement_perturbation_Most_important_features.csv')
        #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'variableDeclare_perturbation_Most_important_features.csv')
        #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'multiLine_comment_perturbation_Most_important_features.csv')
        #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'singleLine_comment_perturbation_Most_important_features.csv')
        X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'.csv')
        Y_test_df = pd.read_csv(data_name+'/data_splits/y_test_project_group'+str(project_group)+'.csv')'''
        X_train_df, Y_train_df, X_valid_df, Y_valid_df, X_test_df, Y_test_df = read_data(data_name_dir, project_group)

        #print(len(X_test_df))
        #print(len(Y_test_df))
        #print(X_test_df)
        #X_test_df.to_csv('XX.csv', index=False)
        #test_name_at_83 = X_test_df.iloc[83]["full_code"]  # Replace "test_name" with the actual column name

        #print(test_name_at_83)
        #exit()
    
        X_test = X_test_df['full_code']
        y_test = Y_test_df['category']
    
        X_train = X_train_df['full_code']
        y_train = Y_train_df['category']
   
        #Y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
    
        #Y_train.columns = ['which_tests']
        y_test.columns = ['which_tests']
    
        # convert labels of train, validation and test into tensors
        #train_y = torch.tensor(Y_train['which_tests'].values)
        test_y = torch.tensor(y_test['which_tests'].values)
    
        # create data_loaders for train and validation dataset
        batch_size = 1

        model = auto_model
        
         
        if ml_technique == "qwen":
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_qwen(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)
                #print('***************** All preds=')
                #print(preds)
        
        elif ml_technique == "gemma7b":
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_gemma7b(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)
                #print('***************** All preds=')
                #print(preds)

        elif ml_technique == "gemma2b":
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_gemma2b(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)

        elif ml_technique == "codegemma":
            #model_name, tokenizer, auto_model = codegemma7b_model_define()
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_codegemma(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)

        elif ml_technique == "llama3_8b":
            #model_name, tokenizer, auto_model = llama3_8b_model_define()
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_llama3_8b(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)
        elif ml_technique == "deep_seek_coder":
            #model_name, tokenizer, auto_model = deep_seek_coder_model_define()
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_deep_seek_coder(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)
        elif ml_technique == "codellama":
            with torch.no_grad():
                preds, category_token_map, top_tokens_per_test = give_test_data_in_chunks_codellama(X_test, tokenizer, model, batch_size, device, project_group, test_y.numpy(), ml_technique)
        else:
            print('no model name found')
        predictions_per_project_group[f"Fold_{project_group}"] = preds
        tokens_per_project_group[f"Fold_{project_group}"] = top_tokens_per_test
        ground_truth_per_project_group[f"Fold_{project_group}"] = test_y
        Org_test_per_project_group[f"Fold_{project_group}"] = X_test

        # Merge tokens from the current project_group into the global category-token map
        for category, tokens in category_token_map.items():
            if category not in global_category_token_map:
                global_category_token_map[category] = []  # Initialize list if category not present
            global_category_token_map[category].extend(tokens)  # Append tokens from current project_group

        cr=classification_report(test_y, preds)
        #print(type(cr))
        parse_cr(cr, technique, str(project_group))
    
        with open(where_data_comes+"-result/classification_report_"+str(project_group)+"project_groups_"+str(project_group), "a") as file:
            file.write("Fold="+str(project_group)+"\n")
            file.write(cr)
            file.write("\n")
    
        cm = confusion_matrix(test_y, preds)
        #print(cm)
    	
        with open(where_data_comes+"-result/confusion_matrix_"+str(project_group)+"project_groups_"+str(project_group), "a") as file:
            file.write("Fold="+str(project_group)+"\n")
            file.write(np.array2string(cm))
            file.write("\n")
    
        #tn, fp, fn, tp = confusion_matrix(test_y, preds, labels=[0, 1]).ravel()
        #TN = TN + tn
        #FP = FP + fp
        #FN = FN + fn
        #TP = TP + tp
        #print("delete model")
        del model
        torch.cuda.empty_cache()
    
        #project_group = project_group+1

    #exit()
    #**Merging & Saving is done AFTER the loop**
    '''df_predictions = pd.DataFrame.from_dict(predictions_per_project_group, orient="index").transpose()
    df_tokens = pd.DataFrame.from_dict(tokens_per_project_group, orient="index").transpose()
    
    
    # Rename columns
    df_predictions.columns = [f"Predictions_{col}" for col in df_predictions.columns]
    df_tokens.columns = [f"Tokens_{col}" for col in df_tokens.columns]
    
    # Merge both DataFrames
    df_combined = pd.concat([df_predictions, df_tokens], axis=1)
    
    # Save to CSV
    df_combined.to_csv(where_data_comes+"-result/"+ml_technique+".csv", index=False)    
    print("\nPredictions and tokens saved to llama.csv")'''
    # Initialize empty list to store reshaped data
    reshaped_data = []
    
    # Iterate over each fold
    for fold in predictions_per_project_group.keys():  # Example: "Fold_1", "Fold_2", ...
        preds = predictions_per_project_group[fold]
        tokens = tokens_per_project_group[fold]
        ground_truths = ground_truth_per_project_group[fold]
        org_test = Org_test_per_project_group[fold]
    
        # Iterate over all samples in this fold
        for test_code, pred, token_list, gt in zip(org_test, preds, tokens, ground_truths):
            reshaped_data.append({
                "test_code": test_code,
                "Prediction": pred,
                "Ground_Truth": int(gt.item()) if isinstance(gt, torch.Tensor) else int(gt),
                "Token_List": token_list
            })
    
    # Convert reshaped data to DataFrame
    df_final = pd.DataFrame(reshaped_data)
    
    # Save to CSV
    csv_path = f"{where_data_comes}-result/{ml_technique}.csv"
    df_final.to_csv(csv_path, index=False)
    
    print("\nPredictions and tokens saved to", csv_path)
    #exit()


    top_10_tokens_per_category = {}
    for category, tokens in global_category_token_map.items():
        token_counts = Counter(tokens).most_common(10)  # Get top-10 most frequent tokens
        top_10_tokens_per_category[category] = token_counts  # Store as (token, count) pairs

    # Convert to DataFrame for better visualization
    df_token_per_cat = pd.DataFrame.from_dict(
        {category: dict(tokens) for category, tokens in top_10_tokens_per_category.items()},
        orient="index"
    ).transpose()
    
    #df_token_per_cat = pd.DataFrame.from_dict(top_5_tokens_per_category, orient='index').transpose()
    # Display result
    # Print the result in a readable format
    print("\nTop-10 Tokens Per Category:")
    for category, tokens in top_10_tokens_per_category.items():
        #print(f"Category {category}: {tokens}")
        print(f"Category {category}:")
        for token, count in tokens:
            print(f"  - {token}: {count}")

    #exit()

def initialize_environment(seed_value):
    """Initializes the environment by setting the seed and configuring logging."""
    set_seed(seed_value)  # Set the seed for reproducibility
    setup_logging()  # Setup standardized logging

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    model_weights_path = sys.argv[2]
    results_file = sys.argv[3]
    data_name_dir = sys.argv[4]
    technique = sys.argv[5]
    initialize_environment(42)
    run_experiment(dataset_path, model_weights_path, results_file, data_name_dir, technique)

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import collections
import re
import pandas as pd
import numpy as np

# Define a set of unwanted tokens
stopwords = {"<|im_start|>", "<|im_end|>", ";", "()", "=", "(", ")", ".", ",", "Ċ" }
import torch
import collections
import re
#exclude_tokens = {"@Test", "public", "void", "test"}
#filtered_tokens = [(t, s) for t, s in top_k_tokens if t.lower() not in exclude_tokens]

def clean_token(token):
    """Remove subword artifacts and special characters."""
    token = token.lower().strip()
    token = re.sub(r'^[ġċ]+', '', token)  # Remove leading artifacts
    token = re.sub(r'[^\w\s]', '', token)  # Remove punctuation
    return token

def collect_token_attention_scores(model, tokenizer, model_inputs, outputs, output_category, k=100):
    print("Extracting attention scores...")

    with torch.no_grad():
        outputs = model(**model_inputs, output_attentions=True, return_dict=True)

    # Extract last layer attention
    attentions = outputs.attentions[-1]  # Last layer shape: [batch, num_heads, seq_len, seq_len]
    average_attention = attentions.mean(dim=1)  # Average across heads

    # Debug: Check for NaNs and replace with zero
    if torch.isnan(average_attention).any():
        print("⚠️ NaN detected in attention scores! Replacing with zero.")
        average_attention = torch.nan_to_num(average_attention, nan=0.0)
     
    print(f"🔹 Raw last-layer attention values (before slicing): {average_attention}")

    # Get input tokens
    tokenized_input = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
    print(f"🔹 Tokenized Input: {tokenized_input}")

    # 🔹 Locate the "Test:" marker in tokenized input
    test_start_index = -1
    for i in range(len(tokenized_input) - 2):
        if tokenized_input[i] == 'ĠTest' and (tokenized_input[i + 1] == ':' or tokenized_input[i + 1] == ':Ċ'):
            test_start_index = i + 2  # Move past "Test:"
            break

    if test_start_index == -1:
        print("❌ Error: 'Test:' not found in tokenized input!")
        return []

    # 🔹 Locate "@Test"
    for i in range(test_start_index, len(tokenized_input) - 1):
        if tokenized_input[i] == 'Ġ@' and tokenized_input[i + 1] == 'Test':
            test_start_index = i  # Adjust start index
            break

    if test_start_index == -1:
        print("❌ Error: '@Test' not found in tokenized input!")
        return []

    # 🔹 Find the end of the test method (closing '}')
    test_end_index = -1
    for i in range(test_start_index, len(tokenized_input)):
        if tokenized_input[i] == '}':
            test_end_index = i + 1  # Include '}'
            break

    if test_end_index == -1:
        test_end_index = len(tokenized_input)  # If no '}', take full remaining tokens

    print(f"📌 Test starts at token index: {test_start_index}, ends at: {test_end_index}")

    # Extract only the test method tokens
    relevant_tokens = tokenized_input[test_start_index:test_end_index]
    
    # 🔹 Locate Category Token in the Model Output
    generated_ids = outputs.logits.argmax(dim=-1)  # Get predicted tokens
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    
    print(f"🔹 Generated Tokens: {generated_tokens}")

    # Extract category name from decoded output
    if "Category:" in output_category:
        category_text = output_category.split("Category:")[-1].strip()  # Get everything after "Category:"
        predicted_category = category_text.split()[0]  # Take the first word (e.g., "Async")
    else:
        predicted_category = None
    
    print(f"✅ Extracted Predicted Category: {predicted_category}")

    # Locate first meaningful category token in generated output
    category_token_index = -1
    for i, token in enumerate(generated_tokens):
        if predicted_category and predicted_category.lower() in token.lower():  
            category_token_index = i
            break

    if category_token_index == -1:
        print("❌ Error: Could not locate category token in generated tokens!")
    else:
        print(f"✅ Category Token '{predicted_category}' Found at Index: {category_token_index}")
    
    print(f"✅ average_attention shape: {average_attention.shape}")

    if len(average_attention.shape) == 4:
        # Standard case: [batch, num_heads, seq_len, seq_len]
        category_attention = average_attention[:, :, test_start_index:test_end_index, category_token_index]  

    elif len(average_attention.shape) == 3:
        # If shape is [num_heads, seq_len, seq_len], remove batch dim
        category_attention = average_attention[:, test_start_index:test_end_index, category_token_index]  
    
    elif len(average_attention.shape) == 2:
        # If shape is [seq_len, seq_len], no batch or head dims
        category_attention = average_attention[test_start_index:test_end_index, category_token_index]  
    
    else:
        raise ValueError(f"❌ Unexpected average_attention shape: {average_attention.shape}")
 
    print(f"✅ category_attention shape after slicing: {category_attention.shape}")

    #aggregated_attention = category_attention.mean(dim=(0, 1))  # Aggregate across batch & heads
    aggregated_attention = category_attention.mean(dim=(0, 1)) if len(category_attention.shape) > 2 else category_attention.mean(dim=0)
    # Normalize attention scores
    max_val = aggregated_attention.max()
    if max_val > 0:
        aggregated_attention /= max_val
    else:
        print("⚠️ Max attention is zero, skipping normalization.")

    print(f"✅ aggregated_attention shape after mean: {aggregated_attention.shape}")

    
    # Ensure aggregated_attention is a valid tensor slice
    if len(aggregated_attention.shape) == 1:
        # Shape [seq_len]
        relevant_attention_scores = aggregated_attention.tolist()
    
    elif len(aggregated_attention.shape) == 2:
        # Unexpected: Shouldn't be 2D here
        relevant_attention_scores = aggregated_attention.mean(dim=1).tolist()
    
    else:
        raise ValueError(f"❌ Unexpected aggregated_attention shape: {aggregated_attention.shape}")
    

    common_java_keywords = {"void", "public", "throws", "int", "if", "for"}
    # ✅ Compute Mean & Standard Deviation of Attention
    mean_attention = np.mean(relevant_attention_scores)
    std_attention = np.std(relevant_attention_scores) + 1e-6  # Avoid zero division
    token_scores = {}
    for token, score in zip(relevant_tokens, relevant_attention_scores):
        token = clean_token(token)
        if token and len(token) > 1:
            # 🚀 Final Scaling: Normalize & Adjust Influence
            normalized_score = (score - mean_attention) / std_attention  # Z-score normalization
            influence_score = np.log1p(abs(normalized_score)) * np.sign(normalized_score)
    
            # ✅ Boost meaningful tokens & suppress Java keywords
            token_scores[token] = influence_score * (1.0 if token not in common_java_keywords else 0.4)

    # ✅ Rank by final influence scores
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)


    # Select the top-k tokens
    top_k_tokens = sorted_tokens[:k]

    # Print the top tokens and their attention scores
    print("\n✅ Top-k Tokens Based on Attention Scores (Only from Test Code):")
    for token, score in top_k_tokens:
        print(f"Token: {token}, Attention Score: {score:.6f}")
    exit()
    return [token for token, _ in top_k_tokens]


#def collect_token_attention_scores(model, tokenizer, model_inputs, outputs, test_data, k=100):
#    print("Extracting attention scores...")
#
#    with torch.no_grad():
#        outputs = model(**model_inputs, output_attentions=True, return_dict=True)
#
#    # Extract last layer attention
#    attentions = outputs.attentions[-1]  # Last layer shape: [batch, num_heads, seq_len, seq_len]
#    average_attention = attentions.mean(dim=1)  # Average across heads
#
#    # Debug: Check for NaNs and replace with zero
#    if torch.isnan(average_attention).any():
#        print("⚠️ NaN detected in attention scores! Replacing with zero.")
#        average_attention = torch.nan_to_num(average_attention, nan=0.0)
#
#    # Get input tokens
#    tokenized_input = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
#
#    # 🔹 Locate the test method in tokenized input
#    test_start_index = -1
#    for i in range(len(tokenized_input) - 2):
#        if tokenized_input[i] == 'ĠTest' and tokenized_input[i + 1] == ':' and tokenized_input[i + 2].startswith('Ċ'):
#            test_start_index = i + 3  # Move to where test method starts
#            break
#
#    if test_start_index == -1:
#        print("❌ Error: 'Test:' not found in tokenized input!")
#        return []
#
#    # 🔹 Locate "@Test"
#    for i in range(test_start_index, len(tokenized_input) - 1):
#        if tokenized_input[i] == 'Ġ@' and tokenized_input[i + 1] == 'Test':
#            test_start_index = i  # Adjust start index
#            break
#
#    if test_start_index == -1:
#        print("❌ Error: '@Test' not found in tokenized input!")
#        return []
#
#    # 🔹 Find the end of the test method (closing '}')
#    test_end_index = -1
#    for i in range(test_start_index, len(tokenized_input)):
#        if tokenized_input[i] == '}':
#            test_end_index = i + 1  # Include '}'
#            break
#
#    if test_end_index == -1:
#        test_end_index = len(tokenized_input)  # If no '}', take full remaining tokens
#
#    print(f"📌 Test starts at token index: {test_start_index}, ends at: {test_end_index}")
#
#    # Extract only the test method tokens
#    relevant_tokens = tokenized_input[test_start_index:test_end_index]
#
#    # Compute attention scores **only for test method**
#    input_attention_received = average_attention.sum(dim=2)  # Shape: [batch, seq_len]
#    input_attention_scores = input_attention_received[0, test_start_index:test_end_index]  # Only test method tokens
#
#    # Normalize attention scores
#    max_val = input_attention_scores.max()
#    if max_val > 0:
#        input_attention_scores /= max_val
#    else:
#        print("⚠️ Max attention is zero, skipping normalization.")
#
#    # Aggregate scores for repeated tokens
#    token_scores = collections.defaultdict(float)
#    token_counts = collections.defaultdict(int)
#
#    for token, score in zip(relevant_tokens, input_attention_scores.tolist()):
#        token = clean_token(token)
#        if token and len(token) > 1:
#            token_scores[token] += float(score)
#            token_counts[token] += 1
#
#    # Sort tokens by total attention received (high to low)
#    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
#
#    # Select the top-k tokens
#    top_k_tokens = sorted_tokens[:k]
#
#    # Print the top tokens and their attention scores
#    print("\n✅ Top-k Tokens Based on Attention Scores (Only from Test Code):")
#    for token, score in top_k_tokens:
#        print(f"Token: {token}, Attention Score: {score:.6f}")
#
#    return [token for token, _ in top_k_tokens]



#import torch
#import collections
#
#def collect_token_attention_scores(model, tokenizer, model_inputs, outputs, text, test_code, k=100):
#    print("Extracting attention scores...")
#
#    with torch.no_grad():
#        outputs = model(**model_inputs, output_attentions=True, return_dict=True)
#
#    # Extract last layer attention
#    attentions = outputs.attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]
#    average_attention = attentions.mean(dim=1)  # Average across heads
#
#    # Debug: Check for NaNs and replace with zero
#    if torch.isnan(average_attention).any():
#        print("⚠️ NaN detected in attention scores! Replacing with zero.")
#        average_attention = torch.nan_to_num(average_attention, nan=0.0)
#
#    # 🔹 Tokenize the full input
#    tokenized_input = tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0])
#
#    # 🔹 Tokenize only the test_code to find its first token
#    test_code_tokens = tokenizer.tokenize(test_code)
#    print("\n🔹 Tokenized Full Input:")
#    print(tokenized_input)
#
#    print("\n🔹 Tokenized Test Method:")
#    print(test_code_tokens)
#    exit()
#
#    if not test_code_tokens:
#        print("❌ Error: Test code could not be tokenized!")
#        return []
#
#    first_test_token = test_code_tokens[0]  # Take the first meaningful token
#
#    # 🔍 **Find where the test method starts**
#    try:
#        start_idx = tokenized_input.index(first_test_token)  # Find first test token in input
#    except ValueError:
#        print(f"❌ Error: First test token '{first_test_token}' not found in tokenized input!")
#        return []
#
#    print(f"📌 Test starts at token index: {start_idx}")
#
#    # Compute attention for **only** test method
#    input_attention_received = average_attention.sum(dim=2)  # Shape: [batch, seq_len]
#    input_attention_scores = input_attention_received[0, start_idx:]  # Extract test code region
#
#    # Normalize attention scores
#    max_val = input_attention_scores.max()
#    if max_val > 0:
#        input_attention_scores /= max_val
#    else:
#        print("⚠️ Max attention is zero, skipping normalization.")
#
#    # Extract **only test-method tokens**
#    test_tokens = tokenized_input[start_idx:]
#
#    # Aggregate scores for repeated tokens
#    token_scores = collections.defaultdict(float)
#    token_counts = collections.defaultdict(int)
#
#    for token, score in zip(test_tokens, input_attention_scores.tolist()):
#        token = clean_token(token)
#        if token and len(token) > 1:
#            token_scores[token] += float(score)
#            token_counts[token] += 1
#
#    # Sort tokens by total attention received (high to low)
#    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
#
#    # Select the top-k tokens
#    top_k_tokens = sorted_tokens[:k]
#
#    # Print the top tokens and their attention scores
#    print("\n✅ Top-k Tokens Based on Attention Scores (Only from Test Code):")
#    for token, score in top_k_tokens:
#        print(f"Token: {token}, Attention Score: {score:.6f}")
#
#    return [token for token, _ in top_k_tokens]







def collect_df_from_data(file_name):
    df = pd.read_csv(file_name)
    X_df = df[['full_code']]
    Y_df = df[['category']]
    return X_df, Y_df 

def read_data(data_name, project_group): 
    X_train_df, Y_train_df = collect_df_from_data(data_name+'/data_split/train_set_'+str(project_group)+'.csv')
    X_valid_df, Y_valid_df = collect_df_from_data(data_name+'/data_split/valid_set_'+str(project_group)+'.csv')

    #X_valid_df = pd.read_csv(data_name+'X_valid_set'+str(project_group)+'.csv')
    #Y_valid_df = pd.read_csv(data_name+'Y_valid_set'+str(project_group)+'.csv')

    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'deadcode_perturbation.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'printStatement_perturbation.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'variableDeclare_perturbation.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'multiLine_comment_perturbation.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'singleLine_comment_perturbation.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'.csv') # Without any perturbation in the test data
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'deadcode_perturbation_Most_important_features.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'printStatement_perturbation_Most_important_features.csv')
    #print("file_name=",data_name+'/data_splits/X_test_project_group'+str(project_group)+'variableDeclare_perturbation_Most_important_features.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'variableDeclare_perturbation_Most_important_features.csv')

    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'multiLine_comment_perturbation_Most_important_features.csv')
    #X_test_df = pd.read_csv(data_name+'/data_splits/X_test_project_group'+str(project_group)+'singleLine_comment_perturbation_Most_important_features.csv')
    X_test_df, Y_test_df = collect_df_from_data(data_name+'/test_set_'+str(project_group)+'.csv')
    #X_test_df = pd.read_csv(data_name+'X_test_set_'+str(project_group)+'.csv') # Without any perturbation in the test data
    #Y_test_df = pd.read_csv(data_name+'Y_test_set'+str(project_group)+'.csv')
    return X_train_df, Y_train_df, X_valid_df, Y_valid_df, X_test_df, Y_test_df


def converting_df_to_series(X_train_df, Y_train_df, X_valid_df, Y_valid_df, X_test_df, Y_test_df):
    X_test = X_test_df['full_code']
    y_test = Y_test_df['category']
    #print(y_test.values)
    X_train = X_train_df['full_code']
    y_train = Y_train_df['category']
    
    X_valid = X_valid_df['full_code']
    y_valid = Y_valid_df['category']
    return X_test, y_test, X_train, y_train, X_valid, y_valid


def create_tensor_and_data_loader(tokens_train, tokens_val, tokens_test, batch_size, seed_worker, train_y, val_y):
    tensor_converter = TensorConverter()
    train_seq, train_mask = tensor_converter.convert_train_to_tensors(tokens_train)
    val_seq, val_mask = tensor_converter.convert_val_to_tensors(tokens_val)
    test_seq, test_mask = tensor_converter.convert_test_to_tensors(tokens_test)

    
    data_loader_factory = DataLoaderFactory(batch_size, seed_worker)
    train_dataloader = data_loader_factory.create_train_loader(train_seq, train_mask, train_y)
    val_dataloader = data_loader_factory.create_val_loader(val_seq, val_mask, val_y)
    return train_dataloader, val_dataloader

# balance dataset,converts a 1D Pandas DataFrame into a 2D NumPy array
def sampling(X_train, y_train):
    print(type(X_train))

    # 1. Convert text data into numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Using 5000 features as an example
    x_train = vectorizer.fit_transform(X_train)

    # 2. Apply SMOTE
    smote = SMOTE(sampling_strategy='minority', random_state=49)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # 3. Convert the sparse matrix to dense dataframe (if necessary)
    x_train_df = pd.DataFrame(x_train.toarray(), columns=vectorizer.get_feature_names())
    x_train_series = x_train_df.apply(lambda row: ' '.join(row.map(str)), axis=1)
    return x_train_series, y_train

# convert code into tokens and then vector representation
class Tokenizer:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def tokenize_data(self, texts):
        tokens = self.tokenizer.batch_encode_plus(
            texts.tolist(),
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True)
        return tokens

    def tokenize_training_data(self, train_text):
        return self.tokenize_data(train_text)

    def tokenize_validation_data(self, val_text):
        return self.tokenize_data(val_text)

    def tokenize_test_data(self, test_text):
        return self.tokenize_data(test_text)

# convert vector representation to tensors
class TensorConverter:
    def __init__(self):
        # If there are any initialization parameters common to all conversion methods,
        # they would be set up here. For this simple case, no initialization is needed.
        pass
    
    @staticmethod
    def convert_to_tensors(tokens):
        """Converts tokenized data to tensors."""
        seq = torch.tensor(tokens['input_ids'])
        mask = torch.tensor(tokens['attention_mask'])
        return seq, mask


    def convert_train_to_tensors(self, tokens_train):
        return self.convert_to_tensors(tokens_train)
    
    def convert_val_to_tensors(self, tokens_val):
        return self.convert_to_tensors(tokens_val)
    
    def convert_test_to_tensors(self, tokens_test):
        return self.convert_to_tensors(tokens_test)

#def text_to_tensors(tokens_train, tokens_val, tokens_test):
#    train_seq = torch.tensor(tokens_train['input_ids'])
#    train_mask = torch.tensor(tokens_train['attention_mask'])
#
#    val_seq = torch.tensor(tokens_val['input_ids'])
#    val_mask = torch.tensor(tokens_val['attention_mask'])
#
#    test_seq = torch.tensor(tokens_test['input_ids'])
#    test_mask = torch.tensor(tokens_test['attention_mask'])
#
#    return train_seq, train_mask, val_seq, val_mask, test_seq, test_mask
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch

class DataLoaderFactory:
    def __init__(self, batch_size, seed_worker=None):
        self.batch_size = batch_size
        self.seed_worker = seed_worker
    
    def create_loader(self, data_seq, data_mask, data_y, sampler_type='random'):
        data = TensorDataset(data_seq, data_mask, data_y)
        if sampler_type == 'random':
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size, worker_init_fn=self.seed_worker)
        return dataloader

    def create_train_loader(self, train_seq, train_mask, train_y):
        return self.create_loader(train_seq, train_mask, train_y, 'random')
    
    def create_val_loader(self, val_seq, val_mask, val_y):
        return self.create_loader(val_seq, val_mask, val_y, 'sequential')
    
    def create_test_loader(self, test_seq, test_mask, test_y):
        return self.create_loader(test_seq, test_mask, test_y, 'sequential')



import os
import time
import sys
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, tree
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy as sp
import pickle
import sklearn.metrics as metrics
import torch.nn.utils.prune as prune
import torch.quantization
import torch.quantization as quantization
from codebert_model import BERT_Arch
from cig import CustomIntegratedGradients
from torch.quantization import fuse_modules
from codebert_model import BERT_Arch
from utils import set_seed, setup_logging, seed_worker, train, evaluate, parse_cr, codebert_model_define, init_setup
from data_processing import sampling, Tokenizer, TensorConverter, DataLoaderFactory, read_data, converting_df_to_series
#from renaming_variable import rename_variables_in_code
import torch.backends.cudnn as cudnn
from perturbation import printStatement_insertion, deadcode_insertion, variableRenaming_insertion, multi_comment_insertion, single_comment_insertion
#from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
import matplotlib.pyplot as plt
from interpret import show
from interpret.blackbox import ShapKernel
from functools import partial
#from shap_explainer import flakicat_explain 
from captum.attr import visualization
from selenium import webdriver
import time
import io
from contextlib import redirect_stdout
from IPython.core.display import display, HTML
from captum.attr import visualization as viz
from bs4 import BeautifulSoup
import subprocess
import torch.nn.functional as F
import csv
import string
from detokenization import combine_tokens 
from calculate_most_and_least_imp_tokens import find_most_and_least_imp_tokens

vis_data_records_ig = []

def combination_of_perturbation(X_test, y_test, X_test_df, perturb):
    perturbations = perturb.split("-")
    k = len(perturbations)
    if k == 2:
        X_test_df['full_code']  = combine_perturbation_technique(X_test, y_test, perturbations[0], perturbations[1])
    if k == 3:
        X_test_df['full_code'] = combine_perturbation_technique(X_test, y_test,  perturbations[0], perturbations[1],  perturbations[2])
    if k == 4:
        X_test_df['full_code'] = combine_perturbation_technique(X_test, y_test,  perturbations[0], perturbations[1],  perturbations[2],  perturbations[3])
    if k == 5:
        X_test_df['full_code'] = combine_perturbation_technique(X_test, y_test,  perturbations[0], perturbations[1], perturbations[2], perturbations[3], perturbations[4])

    return X_test_df


def add_attributions_to_visualizer(attributions, tokens, pred_class, pred_logit, label, vis_data_records, test_y, approximation_error):
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions)

    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()
    pred_list = pred_class.tolist() if isinstance(pred_class, np.ndarray) else pred_class
    flat_tokens = tokens[0]
    #convergence_score = attributions.sum()
    assert len(flat_tokens[:len(attributions)]) == len(attributions), "Tokens and attributions length mismatch"
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred_list[0],
                            pred_logit,
                            label,
                            "label",
                            attributions.sum(),       
                            flat_tokens[:len(attributions)],
                            #convergence_score
                            approximation_error
                            ))    

def save_attributions(tokens, attributions, predictions, csv_filepath, ground_truth, pred_logit, test_confidence_score, test_case, project_group):
    flat_tokens = [item for sublist in tokens for item in sublist] if isinstance(tokens[0], list) else tokens
    flat_attributions = attributions.tolist() if isinstance(attributions, torch.Tensor) else attributions 
    #print(flat_tokens)
    pred_logit_converted = pred_logit.tolist() if isinstance(pred_logit, torch.Tensor) else pred_logit
    confidence_score_converted = test_confidence_score.item() if isinstance(test_confidence_score, torch.Tensor) else confidence_score

    data = [
        {'Token': token, 'Attribution': attr, 'Predicted_class': predictions.item(), 'True_class':ground_truth, 'Test_Pred_logit':pred_logit_converted[0], 'Test_confidence_score':confidence_score_converted} 
        for token, attr in zip(flat_tokens, flat_attributions)
        if token != '<pad>'
    ]
    df = pd.DataFrame(data)
    detokenized_df = combine_tokens(df)

    # Sorting tokens based on attribution scores in descending order
    #sorted_df = detokenized_df.sort_values(by="Attribution", ascending=False)
    
    #tokens_str = ";".join(sorted_df["Token"])
    #attributions_str = ";".join(map(str, sorted_df["Attribution"]))
    # Sorting tokens based on attribution scores in descending order and keeping only top 20
    sorted_df = detokenized_df.sort_values(by="Attribution", ascending=False).head(10)
    
    tokens_str = ";".join(sorted_df["Token"])
    attributions_str = ";".join(map(str, sorted_df["Attribution"]))
    #print(tokens_str)
    #print(attributions_str)
    # Prepare the row data
    row_data = {
        "Project_Group": project_group,
        "Test_case": test_case,
        "Tokens(Sorted by Importance)": tokens_str,
        "Attribution_Scores(Sorted)": attributions_str,
        "Predicted_Class": predictions.item(),
        "True_Class": ground_truth,
        "Test_Pred_Logit": pred_logit_converted[0],
        "Test_Confidence_Score": confidence_score_converted
    }
    
    # Convert to DataFrame
    output_df = pd.DataFrame([row_data])
    
    # Save to CSV (append mode, add header if file doesn't exist)
    output_df.to_csv(csv_filepath, index=False, mode='a', header=not os.path.exists(csv_filepath))
    
    #print(f"Data saved to {csv_filepath}")

def calculate_attributions(input_ids, attention_mask, model, integrated_gradients, label):
    #print(f"Before attribution calculation: input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}") 
    embeddings = model.get_embeddings(input_ids) 
    embeddings = embeddings.clone().detach().float().requires_grad_(True)
    attention_mask = attention_mask.clone().detach().float()
    try:
        attributions= integrated_gradients.attribute(inputs=(embeddings, attention_mask), target=label, return_convergence_delta=True)
        approximation_error = None
        #print("Attributions calculated.")
    except RuntimeError as e:
        print(f"from calculate_attribution Attribution calculation failed: {e}")
        attributions = None
    return attributions, approximation_error


def give_test_data_in_chunks(x_test_nparray, tokenizer, model, batch_size, device, project_group, label, Y_test, dataset_category, attributions_dir, calculate_attribution=False):
    max_length = 512
    x_test = pd.DataFrame(x_test_nparray)
    n = 1
    preds_chunks = None
    paired_data = []
    model.eval()
    total_attributions = []
    total_tokens = []
    total_preds = []
    vis_data_records_ig = []
    count = 0
    confidence_scores = []
    token_attribution_df = pd.DataFrame()
    cig = CustomIntegratedGradients(model)
    attribution_csvfile_name = attributions_dir+dataset_category+"_attributions_project_group_"+str(project_group)+".csv"
    #print("attribution_csvfile_name=", attribution_csvfile_name)
    #exit()
    # Check if file exists and delete it
    if calculate_attribution and os.path.exists(attribution_csvfile_name):
        os.remove(attribution_csvfile_name)
        #print(f"Deleted existing file: {attribution_csvfile_name}")

    if os.path.exists(attribution_csvfile_name):
        os.remove(attribution_csvfile_name) 
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        count +=1
        # Ensure x_test_chunk is a DataFrame
        if isinstance(x_test_chunk, pd.Series):
            x_test_chunk = x_test_chunk.to_frame().T
        # Convert DataFrame to list of strings
        test_data = x_test_chunk.iloc[:, 0].tolist() if len(x_test_chunk) > 1 else [x_test_chunk.iloc[0, 0]]
        test_y = Y_test['which_tests'].iloc[g]
        tokens_test = tokenizer.batch_encode_plus(test_data, max_length=max_length, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids']).to(device).long()
        test_mask = torch.tensor(tokens_test['attention_mask']).to(device).long()

        preds_chunk = model(test_seq, test_mask)
        preds_chunk = preds_chunk.detach().cpu().numpy()
        logit_tensor = torch.tensor(preds_chunk, dtype=torch.float)  # Convert to tensor
        probabilities = F.softmax(logit_tensor, dim=1)
        confidence_score = torch.max(probabilities) 
        confidence_scores.append(confidence_score.item()) 

        pred_class = np.argmax(preds_chunk, axis=1)
        pred_logit = preds_chunk[0, pred_class]
        total_preds.append(pred_class)
        if calculate_attribution:
            try:
                attributions, approximation_error = calculate_attributions(test_seq, test_mask, model, cig, pred_class.item())

                if attributions is not None:
                    attributions_tensor = attributions[0]
                    attributions_sum = attributions_tensor.sum(dim=2).squeeze(0).detach().cpu().numpy()
                    total_attributions.append(attributions_sum)
                    tokens = [tokenizer.convert_ids_to_tokens(id) for id in test_seq.cpu().numpy()]
                    total_tokens.append(tokens)
                    #Print each token with its corresponding attribution score
                    #print("Token | Attribution Score")
                    #print("-------------------------")
            except RuntimeError as e:
                print(f"from give_test_data_in_chunks Attribution calculation failed: {e}")
                print(f"Input IDs shape: {test_seq.shape}, Attention mask shape: {test_mask.shape}")

            #print(attributions_sum.shape)
            save_attributions(tokens, attributions_sum, pred_class, attribution_csvfile_name, test_y, logit_tensor, confidence_score, test_data, project_group)
         
            add_attributions_to_visualizer(attributions_sum, tokens, pred_class, pred_logit, label, vis_data_records_ig, test_y, approximation_error)
    
    modified_html_content = ""
    '''if calculate_attribution:
        print("CALCULATE ***")
        html_obj = viz.visualize_text(vis_data_records_ig)
        html_content = html_obj.data 
        html_content = html_content.replace('True Label', 'Target Label')

        soup = BeautifulSoup(html_content, 'html.parser')
        rows = soup.find_all('tr')
        header_row = rows[0]
        new_col_header = soup.new_tag('th')
        new_col_header.string = 'Ground Truth'
        #header_row.append(new_col_header)
        header_row.insert(1, new_col_header)

        data_rows_start = 1
        for idx, (row, test_value) in enumerate(zip(rows[data_rows_start::2], Y_test['which_tests']), start=data_rows_start):
            new_col_data = soup.new_tag('td')
            new_col_data.string = str(test_value)
            #print('********Ground-truth=', str(test_value))
            row.insert(1, new_col_data)

        modified_html_content = str(soup)'''
    return total_preds, modified_html_content, attribution_csvfile_name, confidence_scores

# Mapping from perturbation technique names to actual function objects
function_map = {
    'deadcode_insertion': deadcode_insertion,
    'printStatement_insertion': printStatement_insertion,
    'variableRenaming_insertion': variableRenaming_insertion,
    'multi_comment_insertion': multi_comment_insertion, 
    'single_comment_insertion': single_comment_insertion
    # Add other mappings here
}

def combine_perturbation_technique(X_test, y_test, perturb1, perturb2, perturb3="", perturb4="", perturb5=""):
    perturbation_array = [perturb1, perturb2, perturb3, perturb4, perturb5] 
    
    # Filter out empty values and calculate the length
    non_empty_perturbations = [p for p in perturbation_array if p]
    num_perturbation = len(non_empty_perturbations) # non-empty pertubation
    
    print(f"Length of non-empty perturbations: {num_perturbation}")

    for perturbation_tech in perturbation_array:
        if perturbation_tech:
            meth_name = perturbation_tech+"_insertion"
            if meth_name in function_map:
            #print(meth_name)
                func = function_map[meth_name]
                X_test = func(X_test, y_test, num_perturbation="multi_perturb_"+str(num_perturbation))
            else:
               print('No function mapped.......')
            #print('***I AM CALLING PERTURBATION****')
     
    return X_test


def boosting_noisy_data_for_train(X_train, y_train, project_group, data_type="train"): 
    # Assuming X_train and y_train are pandas Series and converting them to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Separate data: Category 5 (untouched) & Others (to be perturbed)
    non_category5_indices = np.where(y_train != 5)[0]  # Indices where category is NOT 5
    category5_indices = np.where(y_train == 5)[0]  # Indices where category is 5

    # Get separate subsets
    X_train_non_cat5 = X_train[non_category5_indices]
    y_train_non_cat5 = y_train[non_category5_indices]
    X_train_cat5 = X_train[category5_indices]
    y_train_cat5 = y_train[category5_indices]

    indices = np.random.permutation(len(X_train_non_cat5))
    print(len(X_train_non_cat5))
    split_size = len(X_train_non_cat5) #// 2  # 100% splits 
    
    # List of perturbation functions
    perturbation_functions = [
        "deadcode_perturbation",
        "printStatement_perturbation",
        "variableDeclare_perturbation",
        "multiLine_comment_perturbation",
        "singleLine_comment_perturbation"
    ]

    # Mapping of perturbation names to actual functions
    perturbation_func_map = {
        "deadcode_perturbation": deadcode_insertion,
        "printStatement_perturbation": printStatement_insertion,
        "variableDeclare_perturbation": variableRenaming_insertion,
        "multiLine_comment_perturbation": multi_comment_insertion,
        "singleLine_comment_perturbation": single_comment_insertion
    }

    # Create 5 non-overlapping splits
    X_train_splits = []
    y_train_splits = []
    # **Step 1: Add original data before perturbation**
    X_train_splits.append(pd.DataFrame(X_train, columns=['full_code']))  # Original X_train
    y_train_splits.append(pd.DataFrame(y_train, columns=['category']))  # Original y_train             
    print('len_df=', len(y_train))
    print(y_train_splits[-1].head())

    print('split_size=', split_size)
    for i in range(5):
        # Randomly select 50% of non-category-5 data (without replacement)
        sampled_indices = np.random.choice(len(X_train_non_cat5), size=split_size, replace=False)

        X_split = X_train_non_cat5[sampled_indices]
        y_split = y_train_non_cat5[sampled_indices]

        # Convert to DataFrame
        X_split_df = pd.DataFrame(X_split, columns=['full_code'])
        y_split_df = pd.DataFrame(y_split, columns=['category'])

        # Select perturbation function
        perturbation_name = perturbation_functions[i]
        perturbation_func = perturbation_func_map[perturbation_name]
        print(f"Applying {perturbation_name} to 50% of non-category-5 data")

        # Apply perturbation only on non-category-5 data
        X_split_modified = perturbation_func(X_split_df['full_code'], y_split_df['category'])

        # Store modified data
        X_train_splits.append(pd.DataFrame(X_split_modified, columns=['full_code']))
        y_train_splits.append(pd.DataFrame(y_split, columns=['category']))

    print('******After Orig=',len(X_train_splits))
    # Merge back into the original data
    X_train_augmented = pd.concat(X_train_splits, axis=0).reset_index(drop=True)
    y_train_augmented = pd.concat(y_train_splits, axis=0).reset_index(drop=True)

    X_train_augmented.to_csv("XXX", index=False)
    #print('***=',len(X_train_augmented), len(y_train_augmented ))
    combined_df = pd.concat([X_train_augmented, y_train_augmented], axis=1)

    os.makedirs(f"FlakyLens_Categorization_PerProject-Data/data_split/with_50_percent_noisy_data", exist_ok=True)
    output_csv_file = "FlakyLens_Categorization_PerProject-Data/data_split/with_50_percent_noisy_data/"+data_type+"_set_"+str(project_group)+"_with_noisy_data_excluding_nonflakyCat.csv"
    combined_df.to_csv(output_csv_file, index=False)
    df = pd.read_csv(output_csv_file)
    print("Final df, len=", len(df))
    #y_train_augmented.to_csv(y_output_csv_file, index=False)


def run_experiment(dataset_path, model_weights_path, calculate_attribution, data_name, technique, perturbation):
    device, ml_technique, dataset_category, output_layer, where_data_comes = init_setup(technique, data_name)
    model_name, tokenizer, auto_model = codebert_model_define()

    execution_time = time.time()
    TN = FP = FN = TP = 0
    
    prediction_time_codebert = 0
    categories = 1 # 5 if attribution
    if os.path.exists("../results_per_project/per_Category_Evaluation_BERT-FlakyLens.txt"):
        os.remove("../results_per_project/per_Category_Evaluation_BERT-FlakyLens.txt")
    
    # Get all train-test file pairs from the directory
    train_files = sorted([f for f in os.listdir(data_name_dir) if f.startswith("train_") and f.endswith(".csv")])
    test_files = sorted([f for f in os.listdir(data_name_dir) if f.startswith("test_") and f.endswith(".csv")])

    project_group = 0
    all_predictions = []

    attributions_dir="Attributions_scores/"
    if calculate_attribution == "calculate_attribution_True":
        calculate_attribution = True

        if not os.path.exists(attributions_dir):
            os.mkdir(attributions_dir)
    else:
        calculate_attribution = False

    if calculate_attribution:
        categories = 1 # it will be 6, keeping one to ignore the outer for loop
    for label in range(categories):
        # Ensure each train file has a corresponding test file
        assert len(train_files) == len(test_files), "Mismatch between train and test files"

        for train_file, test_file in zip(train_files, test_files):
            project_group +=1
            if project_group == 5:
                break
            if os.path.exists("FlakyLens_Per_Project_Categorization-result/score_project_group"+str(project_group)+"_Class.txt"):
                os.remove("FlakyLens_Per_Project_Categorization-result/score_project_group"+str(project_group)+"_Class.txt")
            fit_time=0
            bert_flag=0
            total_execution_time = 0
            feature_extraction_time=0
            #total_execution_time_for_feature_extraction = 0
            print(" NOW IN FOLD NUMBER", project_group)
       
            X_train_df, Y_train_df, X_valid_df, Y_valid_df, X_test_df, Y_test_df = read_data(data_name, project_group)
        
            #X_test_df.columns = ['test_code']
            #Y_test_df.columns = ['label']
        
            X_test, y_test, X_train, y_train, X_valid, y_valid = converting_df_to_series(X_train_df, Y_train_df, X_valid_df, Y_valid_df, X_test_df, Y_test_df)
            if "-" in perturbation:
                X_test_df = combination_of_perturbation(X_test, y_test, X_test_df, perturbation) #permutation_count)

            #=============================Adversarial Attack in the test data========================
            #Most or Least
            if perturbation == "deadcode_perturbation":
                print("**** deadcode_perturbation ***")
                X_test_df['full_code'] = deadcode_insertion(X_test, y_test, feature_types="Least_Imp") 
                #perturbation = "deadcode_perturbation"

            elif perturbation == "printStatement_perturbation":
                X_test_df['full_code'] = printStatement_insertion(X_test, y_test, feature_types="Least_Imp")
                #perturbation = "printStatement_perturbation"

            elif perturbation == "variableDeclare_perturbation":
                #X_test_df['full_code'] = variableRenaming_insertion(X_test, y_test, feature_types="Least_Imp")
                #print("I AM HERE")
                print("Before variableRenaming_insertion")
                print(X_test.head())
                perturbed_code = variableRenaming_insertion(X_test, y_test, feature_types="Least_Imp")
                print("After variableRenaming_insertion")
                print(perturbed_code.head())
                X_test_df['full_code'] = perturbed_code

                #perturbation = "variableDeclare_perturbation"

            elif perturbation == "multiLine_comment_perturbation":
                X_test_df['full_code'] = multi_comment_insertion(X_test, y_test, feature_types="Least_Imp")
                #perturbation = "multiLine_comment_perturbation"
                #print("perturbation **")
                #exit()

            elif perturbation == "singleLine_comment_perturbation":
                X_test_df['full_code'] = single_comment_insertion(X_test, y_test, feature_types="Least_Imp")
                #perturbation = "singleLine_comment_perturbation"

            new_file_path = data_name+'/X_test_project_group'+str(project_group)+perturb+'_Most_important_features.csv'
            print('new_file_path=',new_file_path)
            X_test_df.to_csv(new_file_path, index=False)

            # After perturbation, update X_test to use the perturbed code for prediction
            #X_test = X_test_df['full_code']
            print("Perturbed X_test shape:", X_test.head(10))
            #exit()
            #print(new_file_path)
            #print('X_train=', str(len(X_train)), ',y_train=',str(len(y_train)))
            #boosting_noisy_data_for_train(X_train, y_train, project_group)
            #exit()

            ###FOR MAKING PERTURBED TRAINING DATA
            #X_train_df['full_code'] = deadcode_insertion(X_train, y_train) 
            #perturbation = "deadcode_perturbation"
            #X_train_df['full_code'] = printStatement(X_train, y_train)
            #perturbation = "printStatement_perturbation"
            #X_train_df['full_code'] = variableRenaming_insertion(X_train, y_train)
            #perturbation = "variableDeclare_perturbation"


            #X_train_df['full_code'] = multi_comment_insertion(X_train, y_train)
            #perturbation = "multiLine_comment_perturbation"

            #X_train_df['full_code'] = single_comment_insertion(X_train, y_train)
            #perturbation = "singleLine_comment_perturbation"
            #new_file_path = data_name+'/data_splits/X_train_project_group'+str(project_group)+perturbation+'.csv'
            #X_train_df.to_csv(new_file_path, index=False)

            ###FOR MAKING PERTURBED Validation DATA
            #boosting_noisy_data_for_train(X_valid, y_valid, project_group) 
            #X_valid_df['full_code'] = deadcode_insertion(X_valid, y_valid) 
            #perturbation = "deadcode_perturbation"
            #X_valid_df['full_code'] = printStatement(X_valid, y_valid)
            #perturbation = "printStatement_perturbation"
            #X_valid_df['full_code'] = find_variable_declarations(X_valid, y_valid)
            #perturbation = "variableDeclare_perturbation"

            #X_valid_df['full_code'] = multi_comment_insertion(X_valid, y_valid)
            #perturbation = "multiLine_comment_perturbation"

            #X_valid_df['full_code'] = single_comment_insertion(X_valid, y_valid)
            #perturbation = "singleLine_comment_perturbation"
            #new_file_path = data_name+'/data_splits/X_valid_project_group'+str(project_group)+perturbation+'.csv'
            #X_valid_df.to_csv(new_file_path, index=False)

            tokenizer_instance = Tokenizer(tokenizer)
            tokens_train = tokenizer_instance.tokenize_training_data(X_train)
            tokens_test = tokenizer_instance.tokenize_test_data(X_test)

            Y_train = pd.DataFrame(y_train)
            y_test = pd.DataFrame(y_test)
        
            Y_train.columns = ['which_tests']
            y_test.columns = ['which_tests']
        
            # convert labels of train, validation and test into tensors
            train_y = torch.tensor(Y_train['which_tests'].values)
            test_y = torch.tensor(y_test['which_tests'].values)
            #print('Original Labels')
            #print(test_y.numpy())
        
            #train_seq, train_mask, test_seq, test_mask = text_to_tensors(tokens_train, tokens_test)

            tensor_converter = TensorConverter()
            train_seq, train_mask = tensor_converter.convert_train_to_tensors(tokens_train)
            test_seq, test_mask = tensor_converter.convert_test_to_tensors(tokens_test)

            # create data_loaders for train and validation dataset
            #train_dataloader = data_loaders(train_seq, train_mask, train_y)
            batch_size = 1

            data_loader_factory = DataLoaderFactory(batch_size, seed_worker)
            train_dataloader = data_loader_factory.create_train_loader(train_seq, train_mask, train_y)
        
            model = BERT_Arch(auto_model, output_layer)
        
            
            #print(model_weights_path+str(project_group)+'_with_noisy_train_data.pt') 
            '''exit()
            model.load_state_dict(torch.load(model_weights_path+str(project_group)+'.pt'))'''

            #model.load_state_dict(torch.load(model_weights_path+str(project_group)+'_With_noisy_train_data.pt'))
            #model.load_state_dict(torch.load(model_weights_path+str(project_group)+'_Only_noisy_train_1024_data.pt'))

            #print(model_weights_path+'_project_group_'+str(project_group)+'.pt')
            model.load_state_dict(torch.load(model_weights_path+'_project_group_'+str(project_group)+'.pt'))

            model.to(device)
            model.eval()
            start_time = time.time()
            bert_flag=1
            with torch.no_grad():
                preds, html_content, attribution_csvfile_name, confidence_scores = give_test_data_in_chunks(X_test, tokenizer, model, batch_size, device, project_group, label, y_test, dataset_category, attributions_dir, calculate_attribution)
            y_true_values = y_test['which_tests'].values

            preds_to_save = [
                pred[0] if isinstance(pred, (list, tuple)) and len(pred) == 1 else 
                pred.item() if isinstance(pred, (np.ndarray, torch.Tensor)) and pred.size == 1 else 
                pred 
                for pred in preds
            ]
            
            if os.path.exists(attribution_csvfile_name): # Mainly needed to have the important token-list for each test
                token_attribution_df = pd.read_csv(attribution_csvfile_name)
                #print('token_len=',len(token_attribution_df))
                if token_attribution_df.empty:
                    print("token_attribution_df is empty! Skipping token extraction.")
                    token_list = [""] * len(X_test)  # Assign an empty list to prevent errors
                else:
                    if "Tokens(Sorted by Importance)" in token_attribution_df.columns:
                        token_list = token_attribution_df["Tokens(Sorted by Importance)"].to_list()
                    else:
                        print("Column 'Tokens(Sorted by Importance)' not found in token_attribution_df. Using an empty list.")
                        token_list = []
 
                # Prepare the data
                data = {
                    "project_group": [project_group] * len(X_test),  # Track iteration number
                    "test_code": X_test,
                    "Prediction": preds_to_save,
                    "Ground_Truth": y_true_values,
                    "Token_List": token_list
                }
                
                #print(f"X_test length: {len(X_test)}")
                #print(f"token_attribution_df empty? {token_attribution_df.empty}")
                #print(f"Lengths -> X_test: {len(X_test)}, preds_to_save: {len(preds_to_save)}, y_true_values: {len(y_true_values)}, token_list: {len(token_list)}")
                # Create a DataFrame
                prediction_df_to_save = pd.DataFrame(data)
                all_predictions.append(prediction_df_to_save)

            if calculate_attribution: 
                print(f"Saving to: {where_data_comes}-result/Finetuned_Result_with_tokens.csv")
                #print("Data being saved:")
                #print(prediction_df_to_save)
            
                if not prediction_df_to_save.empty:
                    prediction_df_to_save.to_csv(where_data_comes+"-result/"+"Finetuned_Result_with_tokens.csv", mode="a", index=False)
                    print('Successfully saved!')
                else:
                    print("Nothing to save! DataFrame is empty.")

	    	
            preds_array = np.array(preds)
            arr_reshaped = preds_array.reshape(1, -1)
             # Convert the reshaped array to a string with desired formatting
            arr_str = ', '.join(map(str, arr_reshaped[0]))
            result_str = f"project_group{project_group}: {arr_str}"

            original_values_list = Y_test_df['category'].tolist()
            original_y_test = ', '.join(map(str, original_values_list))
            original_y_test_str = f"Original: {original_y_test}"
            
            if calculate_attribution == "calculate_attribution_True":
                file_path = os.path.abspath(where_data_comes+"-result/"+perturbation+"_.txt")
                print('file_path=', file_path)
                print("Current Working Directory:", os.getcwd())


                #with open(where_data_comes+"-result/"+perturbation+".txt" , 'a') as file:
                with open(file_path, 'a') as file:
                    file.write(original_y_test_str)
                    file.write('\n')
                    file.write(result_str)
                    file.write('\n')
            feature_extraction_time = time.time() - start_time

            y_test_for_cr = np.array(y_test).astype(int)
            preds_for_cr = np.array(preds).astype(int)

            #all_classes = list(range(max(max(y_test_for_cr), max(preds_for_cr)) + 1))
            cr=classification_report(y_test, preds, zero_division=0)
            #print(cr)
            #exit()
            category_dict=parse_cr(cr, technique, str(project_group))
            
            #print(type(cr))
            with open(where_data_comes+"-result/"+technique+"_classification_report.txt", "a") as file:
                file.write(technique+",Fold="+str(project_group)+"\n")
                file.write(cr)
                file.write("\n")
                #file.write(","+str(total_execution_time))
        
            cm = confusion_matrix(y_test, preds)
        	
            with open(where_data_comes+"-result/"+technique+"_confusion_matrix_val.txt", "a") as file:
                file.write(technique+",Fold="+str(project_group)+"\n")
                file.write(np.array2string(cm))
                file.write("\n")
            
            del model
            #print("delete model")
            torch.cuda.empty_cache()

        if all_predictions:
            final_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
            save_path = f"{where_data_comes}-result/Finetuned_Result_with_tokens.csv"
            final_predictions_df.to_csv(save_path, index=False)
            
            print(f"Successfully saved all predictions to: {save_path}")
        #else:
        #    print("No data to save!")
  
        with open("../results/per_Category_Evaluation_"+ml_technique+".txt", "w") as file:
            file.write("")
        for cls in range(output_layer):
            cls=str(cls)
            #print(cls)
            avg_category_dict =  [(category_dict[cls][idx] /category_dict[cls][-1]) for idx in range(3)] 
            #print(avg_category_dict)
            with open("../results/per_Category_Evaluation_"+ml_technique+".txt", "a") as file:
                file.write(cls+":" + str(avg_category_dict))
                file.write("\n")
    if calculate_attribution:
        find_most_and_least_imp_tokens()    

def initialize_environment(seed_value):
    """Initializes the environment by setting the seed and configuring logging."""
    set_seed(seed_value)  # Set the seed for reproducibility
    setup_logging()  # Setup standardized logging

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    model_weights_path = sys.argv[2] #"../results_per_project/So-Far-Good-Models/per_project_model_weights_on__dataset" #sys.argv[2]
    calculate_attribution = sys.argv[3]
    data_name_dir = sys.argv[4]
    technique = sys.argv[5]
    #initialize_environment(42)
    perturb = ""
    if len(sys.argv) > 6: #Because the argv starts with the 0 index (0 index is for the script itself)
        perturb = sys.argv[6]
    run_experiment(dataset_path, model_weights_path, calculate_attribution, data_name_dir, technique, perturb)

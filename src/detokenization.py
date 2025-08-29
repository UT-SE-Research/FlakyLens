import pandas as pd
import string
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def is_special_character(token):
    # This function checks if the token is made up only of punctuation or is one of the special tokens
    special_tokens = {'<s>', '</s>', '@', 'Ċ', '=', '()', '.', ';', ',', 'test', 'Test', 'public', 'void', 'private', 'Token', 'not', 'even', 'for', 'To', 'Of', 'With',  'is', 'String', 'int', 'Epoch', 'Assert', 'Id', 'assert', 'At', 'Is', 'ases', 'If', 'Epoch', 'Override', 'And', 'then', 'try', 'throws', 'Integer', 'Person', 'Assert', 'Assert', 'ĊĊ'}
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    return (
        len(token) == 1
        or all(char in string.punctuation for char in token)
        or token in special_tokens
        or is_number(token)
    )
    #return (len(token) == 1 or all(char in string.punctuation for char in token) or token in special_tokens)

def set_new_current_word(token, attribution, pred_class, true_class, pred_logit, confidence_score):
    current_word = token
    current_attribution = float(attribution)
    current_predicted_class = pred_class[0] if isinstance(pred_class, list) else pred_class
    current_true_class = true_class[0] if isinstance(true_class, list) else true_class
    current_pred_logit = pred_logit[0] if isinstance(pred_logit, list) else pred_logit
    current_confidence_score = confidence_score[0] if isinstance(confidence_score, list) else confidence_score
    #print(current_confidence_score)    
    return current_word, current_attribution, current_predicted_class, current_true_class, current_pred_logit, current_confidence_score

def combine_tokens(df):
    combined_df = pd.DataFrame(columns=['Token', 'Attribution', 'Predicted_class', 'True_class', 'Test_Pred_logit', 'Test_confidence_score'])

    current_word = ""
    current_attribution = 0.0
    current_predicted_class = None
    current_true_class = None
    current_pred_logit = None
    current_confidence_score = None

    for _, row in df.iterrows():
        token = row['Token']
        attribution = row['Attribution']
        pred_class = row['Predicted_class']
        true_class = row['True_class']
        pred_logit = row['Test_Pred_logit']
        test_confidence_score = row['Test_confidence_score']
        pred_logit_str = str(pred_logit)  # Convert pred_logit to string to handle the list properly

        #print('From for loop=', str(token), str(attribution), str(pred_class))
        # Start a new word when encountering a token starting with 'Ġ' or a special character
        if pd.notna(token) and token.startswith('Ġ') or is_special_character(token):
            # Remove 'Ġ' for tokens that start with it
            token = token[1:] if token.startswith('Ġ') else token
            if current_word:
                combined_df = combined_df.append({
                    'Token': current_word,
                    'Attribution': current_attribution,
                    'Predicted_class': current_predicted_class,
                    'True_class': current_true_class,
                    'Test_Pred_logit': current_pred_logit,
                    'Test_confidence_score': current_confidence_score
                }, ignore_index=True)
                current_word = ""
                current_attribution = 0.0
            
            # If the token is not a special character, start a new word with it
            if not is_special_character(token):
                current_word, current_attribution, current_predicted_class, current_true_class, current_pred_logit, current_confidence_score = set_new_current_word(token, attribution, pred_class, true_class, pred_logit_str, test_confidence_score)

        elif token[0].isupper():
            #print("Upper word**", token)
            current_word_len = len(current_word)
            #print(current_word, ", len=", str(current_word_len))
            if current_word and (not current_word[current_word_len-1].isupper()): #If current_word is non-empty and current_word capital letter na
                combined_df = combined_df.append({
                    'Token': current_word,
                    'Attribution': current_attribution,
                    'Predicted_class': current_predicted_class,
                    'True_class': current_true_class,
                    'Test_Pred_logit': current_pred_logit,
                    'Test_confidence_score': current_confidence_score
                }, ignore_index=True)
                current_word = ""
                current_attribution = 0.0 
                
                current_word, current_attribution, current_predicted_class, current_true_class, current_pred_logit, current_confidence_score= set_new_current_word(token, attribution, pred_class, true_class, pred_logit_str, test_confidence_score)

            elif current_word and  current_word[current_word_len-1].isupper():
                current_word += token
                current_attribution += float(attribution)
                current_predicted_class = pred_class
                current_true_class = true_class 
                current_pred_logit = pred_logit_str
                current_confidence_score = test_confidence_score

            else:
                #print(type(pred_class))
                current_word, current_attribution, current_predicted_class, current_true_class, current_pred_logit, current_confidence_score = set_new_current_word(token, attribution, pred_class, true_class, pred_logit_str, test_confidence_score)
        else:
            # Continue adding to the current word
            current_word += token
            current_attribution += float(attribution)
            current_predicted_class = pred_class 
            current_true_class = true_class
            current_pred_logit = pred_logit_str
            current_confidence_score = test_confidence_score

    # Append the last word if it exists
    if current_word:
        combined_df = combined_df.append({
            'Token': current_word,
            'Attribution': current_attribution,
            'Predicted_class': current_predicted_class,
            'True_class': current_true_class,
            'Test_Pred_logit': current_pred_logit,
            'Test_confidence_score': current_confidence_score
        }, ignore_index=True)

    return combined_df

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], header=None, names=['Token', 'Attribution', 'Predicted_class', 'True_class', 'Test_Pred_logit','Test_confidence_score'], na_values='', keep_default_na=False)
    start_indices = df[df['Token'] == 'Token'].index
    is_header = (df['Token'] == 'Token') & (df['Attribution'] == 'Attribution')
    start_indices = df[is_header].index
    start_indices = list(start_indices) + [len(df)]
    
    blocks = []
    test_count = 1
    outputDir=sys.argv[2]
    dataset_name = sys.argv[3]
    label = sys.argv[4]
    fold = sys.argv[5]
    outputFile = outputDir + dataset_name+"_category_"+str(label)+"_fold_"+str(fold)+".csv"

    count_async=0
    count_concurrent=0
    count_time=0
    count_uc=0
    count_od=0

    if not os.path.exists(outputDir):
        print('Making detokenization dir')
        os.makedirs(outputDir, exist_ok=True)

    heatmap_dir = outputDir+"heatmaps/"+fold+"/"
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir, exist_ok=True)

    if os.path.exists(outputFile):
        print('removing ', outputFile)
        os.remove(outputFile)

    for i in range(len(start_indices) - 1):
        block = df.iloc[start_indices[i]+1:start_indices[i+1]]  # +1 to skip the header row itself
        blocks.append(block)
       
        with open(outputFile, 'a') as f:
            f.write('#Test='+str(test_count))
            f.write('\n')
        new_df = block[['Token', 'Attribution', 'Predicted_class', 'True_class', 'Test_Pred_logit','Test_confidence_score']] 
        combined_df = combine_tokens(new_df)

        print('*** Saving into ', str(outputFile))
        combined_df['Attribution'] = combined_df['Attribution'].astype(float)
        #top_k = int(len(combined_df) * 0.10)
        top_k = 10 #int(len(combined_df) * 0.10)
        sorted_df = combined_df.sort_values(by='Attribution', ascending=False).head(top_k)
        #sorted_df = combined_df.sort_values(by='Attribution', ascending=False)
        sorted_df.to_csv(outputFile, index=False, mode='a')
        test_count +=1
    

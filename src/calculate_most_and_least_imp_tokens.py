import pandas as pd
from collections import defaultdict
import numpy as np

# === Step 1: List all CSV file paths ===
def find_most_and_least_imp_tokens():
    file_paths = [
        "Attributions_scores/FlakyLens_attributions_project_group_1.csv",
        "Attributions_scores/FlakyLens_attributions_project_group_2.csv",
        "Attributions_scores/FlakyLens_attributions_project_group_3.csv",
        "Attributions_scores/FlakyLens_attributions_project_group_4.csv"
    ]
    
    # === Step 2: Load and concatenate all DataFrames ===
    df_list = [pd.read_csv(fp) for fp in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv("XXX.csv", index=False)
    
    # === Step 3: Initialize token attribution dictionary per class ===
    category_tokens = {i: defaultdict(float) for i in range(6)}  # 5 categories: 0–4
    
    # === Step 4: Aggregate scores for rows where prediction == true class ===
    token_frequency = {i: defaultdict(int) for i in range(6)}  # Track frequency per category
    for idx, row in df.iterrows():
        #print("row=", row)
        print("SHANTO ***, idx=", idx)
        true_class = row['True_Class']
        predicted_class = row['Predicted_Class']
        if true_class == predicted_class and true_class in category_tokens:
            confidence_score = row['Test_Confidence_Score']
            pd.DataFrame([row]).to_csv(f"debug/debug_row_{idx}.csv", index=False)
            try:
                tokens = [t.strip() for t in row['Tokens(Sorted by Importance)'].split(';')]
                scores = list(map(float, row['Attribution_Scores(Sorted)'].split(';')))
                if len(tokens) != len(scores):
                    print("Alas, the Issue occured ****")
                    continue
            except Exception:
                continue
            
            # Sum the attribution scores for each token (before multiplying by confidence score)
            temp_token_scores = defaultdict(float)  # Temporary storage to sum the scores for each token
            for token, score in zip(tokens, scores):
                temp_token_scores[token] += score

            # After summing the scores for each token, multiply by the confidence score
            for token, total_score in temp_token_scores.items():
                weighted_attribution_score = total_score * confidence_score  # Multiply sum by confidence score
                category_tokens[true_class][token] += weighted_attribution_score  # Accumulate the weighted score
                token_frequency[true_class][token] += 1  # Count how many tests this token appears in

                if true_class == 0 and token == "sleep" and predicted_class == 0:
                    print(f"debug/debug_row_{idx}.csv")
                    df_raju=pd.read_csv(f"debug/debug_row_{idx}.csv")
                    print(len(df_raju))
                    print(f"sleep: score={total_score},  predicted_class={predicted_class}, category_tokens={category_tokens[true_class][token]}")

            if true_class == 0 and predicted_class == 0:
                print(f"Total: score={total_score},  predicted_class={predicted_class}, category_tokens_score ={category_tokens[true_class]['sleep']}")
            #exit()

    
    # === Step 5: Save per-class attribution summaries ===
    for cat, token_dict in category_tokens.items():
        token_score_df = pd.DataFrame(list(token_dict.items()), columns=['Token', 'Attribution_Score'])
        token_score_df['Frequency'] = token_score_df['Token'].map(token_frequency[cat])
        # Final importance: Attribution_Score * log(1 + Frequency)
        token_score_df['Final_Importance'] = token_score_df['Attribution_Score'] * np.log1p(token_score_df['Frequency'])
        # Split positive and negative attribution scores
        pos_df = token_score_df[token_score_df['Attribution_Score'] >= 0].sort_values(by='Final_Importance', ascending=False)
        neg_df = token_score_df[token_score_df['Attribution_Score'] < 0].sort_values(by='Final_Importance', ascending=True)
        # Concatenate: positives first, then negatives
        final_df = pd.concat([pos_df, neg_df], ignore_index=True)
        output_path = f'Attributions_scores/category_{cat}_attributions.csv'
        final_df.to_csv(output_path, index=False)


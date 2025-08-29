import pandas as pd
import numpy as np
import sys

input_file = sys.argv[1]
# Read the raw file as plain text to inspect its format
with open(input_file, "r") as file:
    raw_lines = file.readlines()

# Display the first few lines to understand its structure
#print(raw_lines[:10])
# Convert the raw lines into a structured format
formatted_lines = ["Fold,Category,Precision,Recall,F1-Score,Support"]  # CSV header

for line in raw_lines:
    # Remove newline and extract components
    line = line.strip()
    fold_category, metrics = line.split(":[" , 1)
    fold, category = fold_category.split(":")
    metrics = metrics.rstrip("]")  # Remove the closing bracket

    # Append the formatted row
    formatted_lines.append(f"{fold},{category},{metrics}")

# Save the formatted CSV file
formatted_csv_path = "formatted_tmp.csv"
with open(formatted_csv_path, "w") as file:
    file.write("\n".join(formatted_lines))

# Load and display the properly formatted CSV
df_formatted = pd.read_csv(formatted_csv_path)
#tools.display_dataframe_to_user(name="Formatted CSV Data", dataframe=df_formatted)

# Compute macro F1-score (average F1-score across all categories without weighting)
macro_f1 = df_formatted.groupby("Fold")["F1-Score"].mean().mean()

# Compute per-fold F1-score averages
fold_f1_scores = df_formatted.groupby("Fold")["F1-Score"].mean()
print('macro_f1=', macro_f1)
balanced_acc = np.average(df_formatted["Recall"], weights=df_formatted["Support"])
#print("balanced_recall=", balanced_acc)

# Compute total support per fold
df_formatted["Weighted F1"] = df_formatted["F1-Score"] * df_formatted["Support"]

# Compute weighted F1-score per fold
fold_weighted_f1 = df_formatted.groupby("Fold").apply(lambda x: x["Weighted F1"].sum() / x["Support"].sum())

# Compute overall weighted F1-score across all folds
weighted_f1 = df_formatted["Weighted F1"].sum() / df_formatted["Support"].sum()

#print("Per-fold weighted F1-scores:\n", fold_weighted_f1)
#print("Overall weighted F1-score:", weighted_f1)

import pandas as pd

# Load the data from the CSV file
df = pd.read_csv('data.csv')

# Extract the relevant columns
precision = df['precision']
recall = df['recall']
f1_score = df['f1_score']
support = df['support']

# Calculate the total support
total_support = support.sum()
results = []
# Prevent division by zero if total_support is zero
if total_support == 0:
    print("Error: Total support is zero. Cannot compute weighted averages.")
else:
    # Calculate the weighted averages
    weighted_precision = (precision * support).sum() / total_support
    weighted_recall = (recall * support).sum() / total_support
    weighted_f1_score = (f1_score * support).sum() / total_support
    #result.append(weighted_precision)
    #result.append(weighted_recall)
    #result.extend([weighted_precision, weighted_recall, weighted_f1_score])
    result_str = f"{weighted_precision},{weighted_recall},{weighted_f1_score}"
    results.append(result_str)
    #print(weighted_precision,",","," , weighted_recall,",", ",",weighted_f1_score)
    #print(weighted_precision,"," , weighted_recall,",",weighted_f1_score)
    #print(f"Weighted Precision: {weighted_precision:.5f}")
    #print(f"Weighted Recall: {weighted_recall:.5f}")
    #print(f"Weighted F1-Score: {weighted_f1_score:.5f}")
#output = "{},{},{}".format(weighted_precision, weighted_recall, weighted_f1_score)
#print(output)
#print(str(result))
#print(','.join(map(str, result)))
print(",".join(results))

#import pandas as pd
#import ast
#import re
#
## Load the data from the CSV file
#import pandas as pd
#import ast
#import re
#
## Load the data from the CSV file without headers
#df = pd.read_csv('data.csv', header=None, names=['raw'])
#
## Function to extract the list from the string using regex to handle inconsistent spacing
#def extract_list_from_string(s):
#    try:
#        # Find the content inside the square brackets
#        match = re.search(r'\[(.*?)\]', s)
#        if match:
#            # Split the content by comma or space
#            content = match.group(1)
#            return [float(x) for x in re.split(r'[,\s]+', content) if x]
#        else:
#            return [0, 0, 0, 0]  # Default value if extraction fails
#    except (ValueError, SyntaxError):
#        print(f"Error extracting list from string: {s}")
#        return [0, 0, 0, 0]  # Default value if extraction fails
#
## Apply the function to extract the list and create new columns
#df['values'] = df['raw'].apply(extract_list_from_string)
#df[['precision', 'recall', 'f1_score', 'support']] = pd.DataFrame(df['values'].tolist(), index=df.index)
#
## Extract the relevant columns
#precision = df['precision']
#recall = df['recall']
#f1_score = df['f1_score']
#support = df['support']
#
## Calculate the total support
#total_support = support.sum()
#
## Prevent division by zero if total_support is zero
#if total_support == 0:
#    print("Error: Total support is zero. Cannot compute weighted averages.")
#else:
#    # Calculate the weighted averages
#    weighted_precision = (precision * support).sum() / total_support
#    weighted_recall = (recall * support).sum() / total_support
#    weighted_f1_score = (f1_score * support).sum() / total_support
#
#    print(f"Weighted Precision: {weighted_precision:.5f}")
#    print(f"Weighted Recall: {weighted_recall:.5f}")
#    print(f"Weighted F1-Score: {weighted_f1_score:.5f}")
#
#
##df = pd.DataFrame(data, columns=['precision', 'recall', 'f1_score', 'support'])
##
### Extract the relevant columns
##precision = df['precision']
##recall = df['recall']
##f1_score = df['f1_score']
##support = df['support']
##
### Calculate the total support
##total_support = support.sum()
##
### Prevent division by zero if total_support is zero
##if total_support == 0:
##    print("Error: Total support is zero. Cannot compute weighted averages.")
##else:
##    # Calculate the weighted averages
##    weighted_precision = (precision * support).sum() / total_support
##    weighted_recall = (recall * support).sum() / total_support
##    weighted_f1_score = (f1_score * support).sum() / total_support
##
##    print(f"Weighted Precision: {weighted_precision:.5f}")
##    print(f"Weighted Recall: {weighted_recall:.5f}")
##    print(f"Weighted F1-Score: {weighted_f1_score:.5f}")
##
## Function to extract the list from the string using regex to handle inconsistent spacing
##def extract_list_from_string(s):
##    try:
##        # Find the content inside the square brackets
##        match = re.search(r'\[(.*?)\]', s)
##        if match:
##            # Split the content by comma or space
##            content = match.group(1)
##            return [float(x) for x in re.split(r'[,\s]+', content) if x]
##        else:
##            return [0, 0, 0, 0]  # Default value if extraction fails
##    except (ValueError, SyntaxError):
##        print(f"Error extracting list from string: {s}")
##        return [0, 0, 0, 0]  # Default value if extraction fails
##
### Apply the function to extract the list and create new columns
##df['values'] = df['raw'].apply(extract_list_from_string)
##df[['precision', 'recall', 'f1_score', 'support']] = pd.DataFrame(df['values'].tolist(), index=df.index)
##
### Extract the relevant columns
##precision = df['precision']
##recall = df['recall']
##f1_score = df['f1_score']
##support = df['support']
##
### Calculate the total support
##total_support = support.sum()
##
### Prevent division by zero if total_support is zero
##if total_support == 0:
##    print("Error: Total support is zero. Cannot compute weighted averages.")
##else:
##    # Calculate the weighted averages
##    weighted_precision = (precision * support).sum() / total_support
##    weighted_recall = (recall * support).sum() / total_support
##    weighted_f1_score = (f1_score * support).sum() / total_support
##
##    print(f"Weighted Precision: {weighted_precision:.5f}")
##    print(f"Weighted Recall: {weighted_recall:.5f}")
##    print(f"Weighted F1-Score: {weighted_f1_score:.5f}")
##df = pd.read_csv('data.csv', header=None, names=['raw'])
##
##print(df)
### Function to extract the list from the string
##def extract_list_from_string(s):
##    try:
##        start = s.index('[')
##        end = s.index(']')
##        return ast.literal_eval(s[start:end+1])
##    except (ValueError, SyntaxError):
##        print(f"Error extracting list from string: {s}")
##        return [0, 0, 0, 0]  # Default value if extraction fails
##
### Apply the function to extract the list and create new columns
##df['values'] = df['raw'].apply(extract_list_from_string)
##df[['precision', 'recall', 'f1_score', 'support']] = pd.DataFrame(df['values'].tolist(), index=df.index)
##
### Extract the relevant columns
##precision = df['precision']
##recall = df['recall']
##f1_score = df['f1_score']
##support = df['support']
##
### Calculate the total support
##total_support = support.sum()
##
### Prevent division by zero if total_support is zero
##if total_support == 0:
##    print("Error: Total support is zero. Cannot compute weighted averages.")
##else:
##    # Calculate the weighted averages
##    weighted_precision = (precision * support).sum() / total_support
##    weighted_recall = (recall * support).sum() / total_support
##    weighted_f1_score = (f1_score * support).sum() / total_support
##
##    print(f"Weighted Precision: {weighted_precision:.5f}")
##    print(f"Weighted Recall: {weighted_recall:.5f}")
##    print(f"Weighted F1-Score: {weighted_f1_score:.5f}")
##
##total_support = sum(row[3] for row in data)
##
##weighted_precision = sum(row[0] * row[3] for row in data) / total_support
##weighted_recall = sum(row[1] * row[3] for row in data) / total_support
##weighted_f1_score = sum(row[2] * row[3] for row in data) / total_support
##
##print(f"Weighted Precision: {weighted_precision}")
##print(f"Weighted Recall: {weighted_recall}")
##print(f"Weighted F1-Score: {weighted_f1_score}")
#
## Load the data from the CSV file
##df = pd.read_csv('data.csv')
##
### Function to safely evaluate the string representation of lists
##def parse_list_from_string(s):
##    try:
##        return ast.literal_eval(s)
##    except (ValueError, SyntaxError):
##        print(f"Error parsing list from string: {s}")
##        return [0, 0, 0, 0]  # Default value if parsing fails
##
### Parse the 'values' column to extract precision, recall, f1_score, and support
##df['values'] = df['values'].apply(parse_list_from_string)
##
### Create separate columns for precision, recall, f1_score, and support
##df[['precision', 'recall', 'f1_score', 'support']] = pd.DataFrame(df['values'].tolist(), index=df.index)
##
### Validate data types and handle missing values
##df['precision'] = df['precision'].astype(float).fillna(0)
##df['recall'] = df['recall'].astype(float).fillna(0)
##df['f1_score'] = df['f1_score'].astype(float).fillna(0)
##df['support'] = df['support'].astype(float).fillna(0)
##
### Extract the relevant columns
##precision = df['precision']
##recall = df['recall']
##f1_score = df['f1_score']
##support = df['support']
##
### Calculate the total support
##total_support = support.sum()
##
### Prevent division by zero if total_support is zero
##if total_support == 0:
##    print("Error: Total support is zero. Cannot compute weighted averages.")
##else:
##    # Calculate the weighted averages
##    weighted_precision = (precision * support).sum() / total_support
##    weighted_recall = (recall * support).sum() / total_support
##    weighted_f1_score = (f1_score * support).sum() / total_support
##
##    print(f"Weighted Precision: {weighted_precision:.5f}")
##    print(f"Weighted Recall: {weighted_recall:.5f}")
##    print(f"Weighted F1-Score: {weighted_f1_score:.5f}")
##

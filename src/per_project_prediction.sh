#Flakify's flaky test prediction using per-project validation
#dataset=$1
dataset_name=$(echo $1 | cut -d'_' -f1)
data_path="../dataset"
results_path="../models"
#dataset_file="${data_path}/${dataset_name}/${dataset_name}_dataset.csv" 
dataset_file="${data_path}/${dataset_name}/${dataset_name}_dataset_with_nonflaky_indented.csv"  
model_weights="${results_path}/per_project_model_weights_on__dataset"
results_file="${results_path}/sklearn_predictor/per_project_validation_results_on_${dataset}_dataset"

#if [[ ! -d "${results_path}/sklearn_predictor" ]]; then
#    mkdir -p "${results_path}/sklearn_predictor"
#fi
#Flakify_cross_validation_results_on_IDoFT_dataset_Flakify.csv
if [[ $2 == "train" ]]; then
    python3 -W ignore Bert_train_per_project.py $dataset_file $model_weights  "FlakyLens_Categorization_PerProject-Data" "BERT-FlakyLens"

elif [[ $2 == "BERT" ]]; then
    if [[ $3 == "" ]]; then
        python3 -W ignore Testing_per_project.py $dataset_file $model_weights "calculate_attribution_False" "FlakyLens_Categorization_PerProject-Data" "$2-$1" #Only prediction task

    elif [[ $3 == "calculate_gradient" ]]; then
        python3 -W ignore Testing_per_project.py $dataset_file $model_weights "calculate_attribution_True" "FlakyLens_Categorization_PerProject-Data" "$2-$1" #$1=FlakyLens, $2=BERT

    else
        python3 -W ignore Testing_per_project.py $dataset_file $model_weights "calculate_attribution_False" "FlakyLens_Categorization_PerProject-Data" "$2-$1" "$3" #$1=FlakyLens, $2=BERT
    fi
#elif [[ $2 == "llama3_8b" ]]; then
#    python3 Testing_llama3_8b_categorization.py $dataset_file $model_weights "${results_file}_llama3_8b.csv" "FlakyLens_Categorization_PerProject-Data" "$2-$1"

elif [[ $2 == "qwen" || $2 == "gemma7b" || $2 == "gemma2b" || $2 == "codegemma" || $2 == "llama3_8b" || $2 == "deep_seek_coder" || $2 == "codellama" ]]; then
    python3 Testing_other_LLMs.py $dataset_file $model_weights "${results_file}_$2.csv" "FlakyLens_Categorization_PerProject-Data" "$2-$1"
fi



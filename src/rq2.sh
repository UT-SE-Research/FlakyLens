export HF_TOKEN=<hf_your_token_here>
currentDir=$(pwd)
bash per_project_prediction.sh FlakyLens "qwen"
cd ../results/scripts
bash per_category_parse_result.sh ../per_Category_Evaluation_qwen-FlakyLens.txt 
rm ../per_Category_Evaluation_BERT-FlakyLens.txt
cd $currentDir

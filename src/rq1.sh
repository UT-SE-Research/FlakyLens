#!/bin/bash
currentDir=$(pwd)
bash per_project_prediction.sh FlakyLens "BERT"
cd ../results/scripts
bash per_category_parse_result.sh ../per_Category_Evaluation_BERT-FlakyLens.txt
rm ../per_Category_Evaluation_BERT-FlakyLens.txt
cd $currentDir

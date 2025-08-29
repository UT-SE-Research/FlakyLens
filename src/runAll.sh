#!/bin/bash
bash per_project_prediction.sh FlakyLens "train"
bash per_project_prediction.sh FlakyLens "BERT" "calculate_gradient"
bash per_project_prediction.sh FlakyLens "BERT"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode_perturbation"
#bash per_project_prediction.sh FlakyLens "BERT" "printStatement_perturbation"
#bash per_project_prediction.sh FlakyLens "BERT" "variableDeclare_perturbation"
#bash per_project_prediction.sh FlakyLens "BERT" "multiLine_comment_perturbation"
#bash per_project_prediction.sh FlakyLens "BERT" "singleLine_comment_perturbation"

#Perturbation in two
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-variableRenaming"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-single_comment"

#bash per_project_prediction.sh FlakyLens "BERT" "printStatement-variableRenaming"
#bash per_project_prediction.sh FlakyLens "BERT" "printStatement-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "printStatement-single_comment"

#bash per_project_prediction.sh FlakyLens "BERT" "variableDeclare-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "variableDeclare-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "multiLine_comment-singleLine_comment"

#Perturbation in three
#k=3
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-variableRenaming"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-variableRenaming-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-variableRenaming-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-multi_comment-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "printStatement-variableRenaming-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "printStatement-variableRenaming-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "printStatement-multi_comment-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "variableRenaming-multi_comment-single_comment"

#k=4
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-variableRenaming-multi_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-variableRenaming-single_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-multi_comment-signle_comment"
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-multi_comment-single_comment"

#k=5
#bash per_project_prediction.sh FlakyLens "BERT" "deadcode-printStatement-variableRenaming-multi_comment-single_comment"




#bash per_project_prediction.sh FlakyLens "codellama"
#bash per_project_prediction.sh FlakyLens "llama3_8b"
#bash per_project_prediction.sh FlakyLens "qwen"
#bash per_project_prediction.sh FlakyLens "gemma7b"
#bash per_project_prediction.sh FlakyLens "gemma2b"
#bash per_project_prediction.sh FlakyLens "codegemma"
#bash per_project_prediction.sh FlakyLens "deep_seek_coder"


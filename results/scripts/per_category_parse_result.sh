if [[ $1 == "" ]]; then
    echo "argument is missing"
    exit
fi
#$1 #../per_Category_Evaluation_BERT-Flakicat.txt ../without_adversarial_per_Category_Evaluation_BERT-Flakicat.txt

for cat in {0..5}
do
    #echo "$cat"
    grep -r ":$cat:" $1 > tmp
    echo "precision,recall,f1_score,support" > data.csv
    sed 's/^[^[]*\[\([^]]*\)\].*/\1/' tmp >> data.csv
    output=$(python3 per_category_parse_result.py)
    #all_results="${all_results}${output}\n"
    all_results="${all_results}${output},"
    #exit
done 
all_results=${all_results%,}
echo "precision,recall,f1(Async),precision,recall,f1(Conc),precision,recall,f1(Time),precision,recall,f1(UC),precision,recall,f1(OD),precision,recall,f1(Non-flaky)"
echo "$all_results"
rm data.csv
rm tmp
#echo -e "$all_results" | sed '$d'
# Print all results in one row
python3 result_to_show_in_paper_for_imbalanced_data.py $1
rm formatted_tmp.csv

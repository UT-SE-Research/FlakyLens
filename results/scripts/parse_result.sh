if [[ $1 == "" ]]; then
    echo "files.txt"
    exit
fi
#x=0.2 # for five folds
x=0.25 #for four folds

echo "Classifier,Precision,Recall,fscore,total_support,feature_extraction_time,inference_time" > "Result.csv"
while read file
do
    echo $file
    total_precision=0.0
    total_recall=0.0
    total_fscore=0.0
    total_support=0
    total_feature_extraction_time=0.0
    total_inference_time=0.0
    while read line
    do
	    if [[ $line =~ ^\# ]]; then
	        continue
	    fi
            ll=$(echo $line | sed 's/\[//g' | sed 's/\]//g')
            echo $ll 
            precision=$(echo $ll | cut -d',' -f2)
            #echo $precision
            recall=$(echo $ll |cut -d',' -f3 )
            f_score=$(echo $ll |cut -d',' -f4 )
            support=$(echo $ll |cut -d',' -f5 )
            #feature_extraction_time=$(echo $ll |cut -d',' -f6 )
            #inference_time=$(echo $ll |cut -d',' -f7 )
            total_precision=$(echo 'scale=3;'$precision+$total_precision | bc)
            total_recall=$(echo 'scale=3;'$recall+$total_recall | bc)
            total_fscore=$(echo 'scale=3;'$f_score+$total_fscore | bc)
            total_support=$(echo 'scale=3;'$support+$total_support | bc)
            #total_feature_extraction_time=$(echo 'scale=3;'$feature_extraction_time+$total_feature_extraction_time | bc)
            #total_inference_time=$(echo 'scale=3;'$inference_time+$total_inference_time | bc)
    done < $file
    avg_precision=$(echo 'scale=3;'$total_precision*$x | bc)
    avg_recall=$(echo 'scale=3;'$total_recall*$x | bc)
    avg_fscore=$(echo 'scale=3;'$total_fscore*$x | bc)
    #avg_feature_extraction_time=$(echo 'scale=3;'$total_feature_extraction_time*$x | bc)
    #avg_inference_time=$(echo 'scale=3;'$total_inference_time*$x | bc)
    #avg_support=$(echo $total_support | bc)
    echo $avg_precision
    echo $avg_recall
    echo $avg_fscore
    echo $total_support
    #echo $avg_feature_extraction_time
    #echo $avg_inference_time

    #echo "$file,$avg_precision,$avg_recall,$avg_fscore,$total_support,$avg_feature_extraction_time,$avg_inference_time" >> "Result.csv"
    echo "$file,$avg_precision,$avg_recall,$avg_fscore,$total_support"
    echo "$file,$avg_precision,$avg_recall,$avg_fscore,$total_support" >> "Result.csv"
done < $1

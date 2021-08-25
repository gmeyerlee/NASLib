predictors=(seminas xgb gcn bohamiann)

experiment_types=(vary_train_size vary_train_size vary_train_size vary_train_size)


start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=60
fi

# folders:
base_file=/home/zabergjg/NASLib/naslib
s3_folder=ptrans
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=transnasbench101
dataset=class_object

# other variables:
trials=20
end_seed=$(($start_seed + $trials - 1))
save_to_s3=false
test_size=200

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}
    python $base_file/benchmarks/create_configs.py --predictor $predictor --experiment_type $experiment_type \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --config_type predictor --search_space $search_space
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/predictors/config\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python $base_file/benchmarks/predictors/runner.py --config-file $config_file
    done
    if [ "$save_to_s3" = true ]
    then
        # zip and save to s3
        echo zipping and saving to s3
        zip -r $out_dir.zip $out_dir 
        python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir --s3_folder $s3_folder
    fi
done

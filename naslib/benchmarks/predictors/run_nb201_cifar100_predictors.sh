<<<<<<< HEAD:naslib/benchmarks/predictors/run_transmacro_classObject.sh
predictors=(xgb bohamiann gp rf mlp)

experiment_types=(single single single single single)
=======
predictors=(gp rf mlp bohamiann xgb)
>>>>>>> 32892c560d27426eea433064aa74d80fff5718ce:naslib/benchmarks/predictors/run_nb201_cifar100_predictors.sh

experiment_types=(single single single single single)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
<<<<<<< HEAD:naslib/benchmarks/predictors/run_transmacro_classObject.sh
base_file=/home/shalag/NASLib/naslib
s3_folder=ptransmacro_hyper
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=transbench_macro
dataset=class_object

# other variables:
trials=5
=======
base_file=/home/zabergjg/NASLib/naslib
s3_folder=p201_nohyper
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=nasbench201
dataset=cifar100

# other variables:
trials=100
>>>>>>> 32892c560d27426eea433064aa74d80fff5718ce:naslib/benchmarks/predictors/run_nb201_cifar100_predictors.sh
end_seed=$(($start_seed + $trials - 1))
save_to_s3=false
test_size=200
train_size_single=100
<<<<<<< HEAD:naslib/benchmarks/predictors/run_transmacro_classObject.sh
max_hpo_time=2000
=======
>>>>>>> 32892c560d27426eea433064aa74d80fff5718ce:naslib/benchmarks/predictors/run_nb201_cifar100_predictors.sh

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    experiment_type=${experiment_types[$i]}
    python $base_file/benchmarks/create_configs.py --predictor $predictor --experiment_type $experiment_type \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
<<<<<<< HEAD:naslib/benchmarks/predictors/run_transmacro_classObject.sh
    --dataset=$dataset --config_type predictor --search_space $search_space --train_size_single $train_size_single --max_hpo_time $max_hpo_time
=======
    --dataset=$dataset --config_type predictor --search_space $search_space --train_size_single $train_size_single
>>>>>>> 32892c560d27426eea433064aa74d80fff5718ce:naslib/benchmarks/predictors/run_nb201_cifar100_predictors.sh
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


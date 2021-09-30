# here we construct the first one-shot algorithm script from outsiders point of view
# what people need is a concrete example of 
# 1. oneshot algorithm (e.g. RandomNAS) (in this case, this script is for RandomNAS)
# 2. Hyperparameter setting 
# 3. List of search spaces
# End of the story

spaces=("nasbenchasr")
# spaces=("nasbench101" "nasbench201" "darts")
algorithm='rsws'


for ((i=0;i<${#spaces[@]};i=i+1)); do
    space=${spaces[$i]}
    config=configs/${algorithm}-${space}.yaml
    python oneshot_runner.py --config-file=${config}
done

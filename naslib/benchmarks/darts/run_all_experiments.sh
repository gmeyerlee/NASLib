#!/bin/bash

optimizers=(darts gdas drnas)
#search_spaces=(nasbench101 darts)
search_spaces=(nasbench201)

#datasets=(cifar10)
datasets=(cifar10 cifar100 ImageNet16-120)

for opt in "${optimizers[@]}"
do
	for search_space in "${search_spaces[@]}"
	do
		for dataset in "${datasets[@]}"
			do
				echo run_exp.sh $opt $search_space $dataset
				sbatch run_exp.sh $opt $search_space $dataset
			done
	done
done

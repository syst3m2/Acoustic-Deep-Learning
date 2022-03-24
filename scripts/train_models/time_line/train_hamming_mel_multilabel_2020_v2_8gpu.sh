#!/bin/bash
#SBATCH --job-name=20-sonar
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=336:00:00
#SBATCH --partition=beards
#SBATCH --output=train_multilabel_mel_2020_v2-8gpu-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/model_output/current_runs/output_mel_multilabel_2020_8gpu_v2' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --model='dev_bnn_model' --mode='train' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=1000 --callbacks='csv_saver, checkpoint, tensorboard, reduce_lr' --num_classes=5 --bnn_type='dropout' \
	           --class_weight=False --data_aug=False --optimizer='sgd' --model_type='multi_label' --duration=30 --channels=4 --learning_rate_start=0.01 \
               --start_date='20200101' --end_date='20201231'


#!/bin/bash
#SBATCH --job-name=sonar
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=168:00:00
#SBATCH --partition=beards
#SBATCH --output=train_multilabel_mel_012019_082021_v4-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/model_output/output_mel_multilabel_cs4921_012019_082021_v4' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --model='dev_bnn_model' --mode='train' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=50 --callbacks='csv_saver, checkpoint, tensorboard, reduce_lr' --num_classes=5 --bnn_type='dropout' \
	           --class_weight=False --data_aug=False --optimizer='sgd' --model_type='multi_label' --duration=30 --channels=4 --learning_rate_start=0.01
               --start_date='20200101' --end_date='20201231'


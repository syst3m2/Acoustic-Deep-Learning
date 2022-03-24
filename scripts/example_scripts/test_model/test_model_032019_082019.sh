#!/bin/bash
#SBATCH --job-name=test_sonar
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=beards
#SBATCH --output=saved-model-032019-082019-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012021_082021/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/model_output/to_compare/output_mel_multilabel_cs4921_032019_082019_v1/' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --saved_model_path='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/model_output/to_compare/output_mel_multilabel_cs4921_032019_082019_v1/checkpoint317--0.88.h5' \
               --model='dev_bnn_model' --mode='saved' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=50 --callbacks='csv_saver, checkpoint, tensorboard, reduce_lr' --num_classes=5 --bnn_type='none' --bnn='True' --bnn_build='True' \
	           --class_weight=False --data_aug=False --optimizer='sgd' --model_type='multi_label' --duration=30 --channels=4 --learning_rate_start=0.01 --test_data_type='data_dir'

date
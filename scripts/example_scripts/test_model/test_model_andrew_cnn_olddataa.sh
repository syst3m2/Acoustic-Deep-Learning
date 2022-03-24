#!/bin/bash
#SBATCH --job-name=andrew_cnn_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --partition=beards
#SBATCH --output=saved-model-andrew-4channel-pred-test--%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/data/kraken/teams/acoustic_data/mars_4channel_multiclass_mar_aug_2019/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/test_models/4-channel-checkpoint-andrew-cnn/predict_test_2/' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --saved_model_path='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/test_models/4-channel-checkpoint-andrew-cnn/4-channel-checkpoint-andrew-0.69.h5' \
               --model='cnn_model' --mode='saved' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_classes=5 --bnn='False' --bnn_build='False' \
	           --class_weight=False --data_aug=False --model_type='multi_class' --duration=30 --channels=4 --test_data_type='data_dir' \
               --dataset='wav'

date
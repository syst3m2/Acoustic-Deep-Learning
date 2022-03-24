#!/bin/bash
#SBATCH --job-name=01_06_20_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5 
#SBATCH --output=saved-model-bnn-testpred-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/data/kraken/teams/acoustic_data/mars_4channel_multiclass_mar_aug_2019/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/test_models/4-channel-dropout-bnn/predict_test/' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --saved_model_path='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/test_models/4-channel-dropout-bnn/checkpoint288--0.81_dropout_4channel_sabrina.h5' \
               --model='dev_bnn_model' --mode='saved' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=50 --num_classes=5 --bnn_type='dropout' --bnn='True' --bnn_build='True' \
	           --class_weight=False --data_aug=False --model_type='multi_class' --duration=30 --channels=4 \
               --dataset='wav'

date
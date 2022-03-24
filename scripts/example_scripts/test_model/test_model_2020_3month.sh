#!/bin/bash
#SBATCH --job-name=07_12_20_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=168:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5 
#SBATCH --output=z_saved-model-2020-pred-3month-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/old_data/multilabel_mel_012019_082021/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/to_compare/output_mel_multilabel_072020_122020_v2/predict_3month/' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --saved_model_path='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/to_compare/output_mel_multilabel_072020_122020_v2/checkpoint495--0.85.h5' \
               --model='dev_bnn_model' --mode='saved' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=50 --num_classes=5 --bnn_type='dropout' --bnn='True' --bnn_build='True' \
	           --class_weight=False --data_aug=False --model_type='multi_label' --duration=30 --channels=4 --test_data_type='data_dir' \
               --start_date='20210101 000000' --end_date='20210401 000000' --callbacks=''

date
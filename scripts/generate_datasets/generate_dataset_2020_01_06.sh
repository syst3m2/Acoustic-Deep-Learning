#!/bin/bash
#SBATCH --job-name=0104_19_datagen
#SBATCH --nodes=1
#SBATCh --cpu-per-task=16
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5 
#SBATCH --output=gen_data_2020_01_06-%j.txt


. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20200612 211054' --end_date '20200701 000000' \
                        --label_type='multilabel'
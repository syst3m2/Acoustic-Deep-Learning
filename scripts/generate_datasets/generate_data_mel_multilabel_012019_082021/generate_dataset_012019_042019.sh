#!/bin/bash
#SBATCH --job-name=0104_19_datagen
#SBATCH --nodes=1
#SBATCh --cpu-per-task=16
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5 
#SBATCH --output=/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/generate_dataset/mel_multilabel_01_04_2019/data_gen_01_04_2019-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python /smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/test_data' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '01/01/2019' --end_date '04/01/2019'


date
#!/bin/bash
#SBATCH --job-name=audio_classifier
#SBATCH --nodes=1
#SBATCh --cpu-per-task=4
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=output_0418_0618-%j.txt
#SBATCH --partition=beards
#SBATCH --gres=gpu



. /etc/profile

module load lang/miniconda3/4.5.12

source activate cs4321

python main_pipeline.py --length=30 --output_data='calibrated-mel' --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' --audio_database='group/mbari/upload/MARSInfo' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/mel_calibrated_052018_082021_v2' --channels=4 --label_range=20 --sample_rate=4000 --start_date '04/06/2018' --end_date '07/01/2018'
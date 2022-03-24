#!/bin/bash
#SBATCH --job-name=audio_classifier
#SBATCH --nodes=1
#SBATCh --cpu-per-task=4
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=output_0721_0821-%j.txt
#SBATCH --partition=beards
#SBATCH --gres=gpu


. /etc/profile

module load lang/miniconda3/4.5.12

source activate cs4321

python main_pipeline.py --length=30 --output_data='calibrated-mel' --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' --audio_database='smallwork/beards/CS4321/acoustic_datasets/database_07_08_2021/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/mel_calibrated_012019_082021' --channels=4 --label_range=20 --sample_rate=4000 --label_type='multiclass' --start_date '07/01/2021' --end_date '08/07/2021'
#!/bin/bash
#SBATCH --job-name=audio_classifier
#SBATCH --nodes=1
#SBATCh --cpu-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=nogpu_output_0120_0320_example-%j.txt
#SBATCH --partition=beards




. /etc/profile

module load lang/miniconda3/4.5.12

source activate cs4321

python main_pipeline.py --length=30 --output_data='mel' --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' --audio_database='smallwork/beards/CS4321/acoustic_datasets/database_10_12_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/mel_032019_082019' --channels=4 --label_range=20 --sample_rate=4000 --start_date '05/01/2019' --end_date '06/01/2019'
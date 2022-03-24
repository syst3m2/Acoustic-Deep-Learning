#!/bin/bash
#SBATCH --job-name=audio_classifier
#SBATCH --nodes=1
#SBATCh --cpu-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5
#SBATCH --output=nogpu_output-%j.txt




. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env

python /smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/utilities/audioClipGenerator.py --label_type=class \
--mode=phys --data_dir='/data/kraken/teams/acoustic_data/MARS_hour_long_4k_4channel_data' \
--target_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/data_test/audio' \
--csv_file='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/data_test/csv/ship_list.csv' \
--multi_label='False'
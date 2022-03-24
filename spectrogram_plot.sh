#!/bin/bash
#SBATCH --job-name=spectrogram_plot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5
#SBATCH --output=ais_plot-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

# First, find a ship or time period you would like to make a plot for. If it's a time period...

python3 main.py  --data_dir='/smallwork/beards/CS4321/acoustic_datasets/test_data/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/plots/ais/' \
               --mode='plot' --plot_type='ais' --data_type='mel' --sample_rate=4000 \
               --channels=4 --start_date='20190505 084727' --end_date='20190505 145937'

date
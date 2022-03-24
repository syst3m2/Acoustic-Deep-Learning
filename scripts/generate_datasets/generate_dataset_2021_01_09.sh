#!/bin/bash
#SBATCH --job-name=0109_21_datagen
#SBATCH --nodes=1
#SBATCh --cpu-per-task=16
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5 
#SBATCH --output=gen_data_2021_01_09-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

srun --pty --nodes=1 --mem=100G --time=14:00:00 --gres=gpu:1 --partition=kraken --nodelist=compute-9-5 bash
module load lang/miniconda3/4.5.12
source activate model_env


python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_04_07_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210426 170910' --end_date '20210501 000000' \
                        --label_type='multilabel'

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210531 062157' --end_date '20210601 000000' \
                        --label_type='multilabel'

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_10_12_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210620 184640' --end_date '20210701 000000' \
                        --label_type='multilabel'

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_10_12_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210806 101730' --end_date '20210810 000000' \
                        --label_type='multilabel'

####################################################################################################


python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210425' --end_date '20210501' \
                        --label_type='multiclass'

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_04_07_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210526' --end_date '20210601' \
                        --label_type='multiclass'



python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_07_10_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210621' --end_date '20210701' \
                        --label_type='multiclass'

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210730' --end_date '20210810' \
                        --label_type='multiclass'


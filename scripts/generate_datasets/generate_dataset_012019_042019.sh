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

srun --pty --nodes=1 --mem=100G --time=24:00:00 --gres=gpu:1 --partition=kraken --nodelist=compute-9-5 bash
srun --pty --nodes=1 --mem=100G --time=24:00:00 --gres=gpu:1 --partition=beards bash
module load lang/miniconda3/4.5.12
source activate model_env


. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

tail -f /proc/<pid>/fd/1

python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/test_data' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20190101' --end_date '20190210' \
                        --label_type='multiclass' &
                        
> 2019_0101_2019_0201.txt &
34967

tail -f /proc/<pid>/fd/1

running
python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20190101' --end_date '20190701' \
                        --label_type='multiclass' >> 2019_01_06.txt &

running
python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_04_07_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20190701' --end_date '20200101' \
                        --label_type='multiclass' >> 2019_07_12.txt &

running
python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_07_10_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20200101' --end_date '20200701' \
                        --label_type='multiclass' >> 2020_01_06.txt &

running
python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_01_04_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20200701' --end_date '20210101' \
                        --label_type='multiclass' >> 2020_07_12.txt &

running
python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_04_07_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210101' --end_date '20210901' \
                        --label_type='multiclass' >> 2021_01_09.txt &



srun --pty --nodes=1 --mem=128G --time=168:00:00 --gres=gpu:1 --partition=kraken --nodelist=compute-9-5 bash
module load lang/miniconda3/4.5.12
source activate model_env
python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_07_10_2020/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multiclass_mel_012019_082021' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210101' --end_date '20210806' \
                        --label_type='multilabel'


date
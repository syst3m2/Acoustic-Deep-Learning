## CS4321 Final Project (Sonar Data Analysis)

Any scripts that are run must be moved into the base directory of this repo.

**To Create a New Dataset**

**Permissions**
To do the following, you must have permissions to access the /group/mbari folder on Hamming for the acoustic data and the Physics AIS Box folder.

**Step 1**: 
If you have access, visit physics Box folder to get AIS .mat files, copy to ais folder

**Step 2**: 
Attach to a compute node, load anaconda, then activate an environment with the dependencies stored in requirements.txt
```
srun --pty --mem=64G --gres=gpu:1080:1 --partition=beards bash
module load lang/miniconda3/4.5.12
conda activate [environment_name] or source activate [environment_name]
pip install -r requirements.txt
```

**Step 3**: 
Run the following commands to install the wavcrawler tools to access the audio data:

```
pip install -e resources/vs_db_maintainer/
pip install -e resources/vs_data_query/
pip install -e resources/sqlite_resources/
pip install -e resources/data_types/
```

**Step 4**: 
This step will be changing soon as the data pipeline will migrate to a MySQL database.

Because sqlite does not support concurrent operations, navigate to the following folder:

```
/group/mbari/upload/MARSinfo
```
and make a copy of the master_index.db for every script you will run concurrently. If only one script is needed, then just make one copy of the database to a location of your choosing.

**Step 5**: 
Get an example script from the scripts folder and run the following to generate a new dataset. This script will skip audio data without corresponding .mat files, since there aren't .mat files for every day

`./generate_dataset.sh --output_folder '/path/to/output/folder' --data_folder '/path/to/data/folder' --ais_folder '/path/to/ais/folder' --label_type 'multilabel' --start_date '06/29/2021' --end_date '06/30/2021'`

you may pass optional parameters as follows

```
--length=30
--output_data='calibrated-mel'
--channels=4 
--label_range=20 
--sample_rate=4000 

```

An example dataset generation script looks like:

```
srun --pty --nodes=1 --mem=100G --time=14:00:00 --gres=gpu:1 --partition=kraken --nodelist=compute-9-5 bash
module load lang/miniconda3/4.5.12
source activate model_env


python main_pipeline.py --length=30 --output_data='mel' \
                        --ais_folder='smallwork/beards/CS4321/acoustic_datasets/ais_052018_082021' \
                        --audio_database='smallwork/beards/CS4321/acoustic_datasets/databases/database_04_07_2019/' \
                        --output_folder='smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2' \
                        --channels=4 --label_range=20 --sample_rate=4000 --start_date '20210426 170910' --end_date '20210501 000000' \
                        --label_type='multilabel'
```


Dataset Options (A dataset can be generated with the following options):

    length: (e.g. 30) number of seconds to generate the audio for, each labeled audio will correspond to this number
    output_data: 'raw', 'calibrated', 'mel', 'calibrated-mel' this will output either raw audio data, 
            calibrated audio data, or the mel spectrogram of the audio data
    ais_folder: absolute file path containing ais .mat files
    audio_database: absolute file path containing audio database
    output_folder: absolute file path for desired data output
    channels: number of channels to have in the resulting dataset
    label_range: Range in kilometers to count as true labels for ships
    sample_rate: Desired sample rate to downsample to, must be less than or equal to 8000
    label_type: Multiclass or multilabel depending on which type of model you want to train
    start_date: The date for which the first audio file should be gathered (format yyyymmdd hhmmss) 
    end_date: The date for which the last audio file should be gathered (format yyyymmdd hhmmss)


To run the script in the background with an active terminal:

```
./example_generate_dataset.sh > output.txt &
```
or submit as a job with

```
sbatch example_generate_dataset.sh
```

To view output from file
```
tail -f output.txt
```

Unfortunately, running the script in the background has issues where it fails occassionally and does not output the data to the .txt file. Since this is the case, I recommend running the script with an active session instead of submitting the script with an sbatch command. To do this, run the example command shown above.

Multiple scripts can be run at the same time to parallelize creating a dataset. 3 Months is the recommended max period for each script (takes approximately 30 hours to execute)

**Additional Info**

Here are some additional ways to get a interactive GPU session on Hamming

```
srun --pty  --partition=beards  --gres=gpu:titanrtx    /bin/bash   # if you want a Titan RTX

srun --pty  --partition=beards  --gres=gpu:1080    /bin/bash   # if you want a 1080 Ti

srun --pty  --partition=beards  --gres=gpu    /bin/bash   # if you’re not picky

srun --pty  --partition=beards  --gres=gpu:2    /bin/bash   # if you need more than 1 GPU

srun --pty  --partition=beards  --gres=gpu:titanrtx:3    /bin/bash   # if you need 3 Titan RTX
```


**To Train a Model**

An example training script is as shown, put in a .sh file and submit as an sbatch.

```
#!/bin/bash
#SBATCH --job-name=20-day
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=11:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5
#SBATCH --output=train_multilabel_01_06_2020_day-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/current_runs/multilabel_2020_day_split' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --model='dev_bnn_model' --mode='train' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=1000 --callbacks='csv_saver, checkpoint, tensorboard, reduce_lr' --num_classes=5 --bnn_type='dropout' \
	           --class_weight=False --data_aug=False --optimizer='sgd' --model_type='multi_label' --duration=30 --channels=4 --learning_rate_start=0.01 \
               --start_date='20200101 000000' --end_date='20200531 000000' --test_data_type='new_split' --shuffle_group='day'
```

**To Continue Training a Model**

To continue training a model, check the output folder and look at the last saved checkpoint. Note the epoch of that checkpoint, then go to the output file (not the log.csv) and find that epoch and note the learning rate output. You will need to change the learning rate start parameter in the below script to the one from the last epoch to continue training. You also need to change the mode to cont-train. Be sure to change the output folder to a new folder or else it will overwrite previous checkpoints.

I have not edited the code to do this, but I learned that you can pass a starting epoch parameter to the model.fit method and that would allow you to continue pointing to the same folder.

To do the performance plots for this, you need to combine the log.csv files into a single one. Then you can use log_format.py to make all the epoch numbers count correctly. Be sure to delete epochs in the log.csv from the previous runs. For example, if you continued training from a checkpoint, the log.csv may have additional checkpoint, but you need to delete these because you continued training from before those epochs.


```
#!/bin/bash
#SBATCH --job-name=20-day
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=11:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5
#SBATCH --output=train_multilabel_01_06_2020_day-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/current_runs/multilabel_2020_day_split_v2' 
                --saved_model_path='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/model_output/current_runs/output_mel_multilabel_cs4921_2019_v2_cont/checkpoint271--0.76.h5' \
                --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --model='dev_bnn_model' --mode='cont-train' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=1000 --callbacks='csv_saver, checkpoint, tensorboard, reduce_lr' --num_classes=5 --bnn_type='dropout' \
	           --class_weight=False --data_aug=False --optimizer='sgd' --model_type='multi_label' --duration=30 --channels=4 --learning_rate_start=0.01 \
               --start_date='20200101 000000' --end_date='20200531 000000' --test_data_type='new_split' --shuffle_group='day'
```

**To Test a Model**

To test a model, be sure the input parameters are the same as the ones from when you trained the model. Because there are different data processing methods, this is important.

```
#!/bin/bash
#SBATCH --job-name=bnn_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=kraken
#SBATCH --nodelist=compute-9-5 
#SBATCH --output=z_saved-model-bnn-day-3month-%j.txt

. /etc/profile
module load lang/miniconda3/4.5.12
source activate model_env
date

python3 main.py --data_dir='/smallwork/beards/CS4321/acoustic_datasets/multilabel_mel_012019_082021_v2/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/current_runs/multilabel_2020_day_split/predict_3month/' --overlap=75 --sample_pts=1024 --mel_bins=128 --batch_size=256 \
               --saved_model_path='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/outputs/model_training/current_runs/multilabel_2020_day_split/checkpoint02--0.54.h5' \
               --model='dev_bnn_model' --mode='saved' --model_input='mfcc' --print_specgram=False --resnet_n=3 --classes='classA, classB, classC, classD, classE' \
               --num_epochs=50 --num_classes=5 --bnn_type='dropout' --bnn='True' --bnn_build='True' \
	           --class_weight=False --data_aug=False --model_type='multi_label' --duration=30 --channels=4 --test_data_type='new_split' --shuffle_group='day' \
               --start_date='20200701 000000' --end_date='20201001 000000' --dataset='tfrecord'

date
```

**To Create Model Plots**

To get the performance plots after training, use the performance_plots.py python script. You need to change the path to the csv file to point at the csv log. If you continued training, as noted in the continue training section, you will need to combine the csv logs.


To make a plot of spectrograms going into the model for training (it makes 20), just change the print_specgrams parameter to True in the train model script. You should see the below as output.

/example_plots/spectrograms_example.png

![Spectrogram Model Input plot](/example_plots/spectrograms_example.png?raw=true "Spectrogram Model Input Plot")



To plot a single spectrogram of a chunk of acoustic data like the below

![Single Spectrogram Plot](/example_plots/spectrogram_example.png?raw=true "Single Spectrogram Plot")

Use the following script as a template

```
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

python3 main.py  --data_dir='/smallwork/beards/CS4321/acoustic_datasets/test_data/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/plots/ais/' \
                --database='/group/mbari/upload/MARSInfo/master_index.db' \
               --mode='plot' --plot_type='spectrogram' --data_type='mel' --sample_rate=4000 \
               --channels=4 --start_date='20190505 084727' --end_date='20190505 084757'

date
```




To get the below output, you can use the below script as a template. This takes a defined time period and makes a prediction and plots the prediction. Do not take too much data, this is only intended for small time periods due to memory constraints. This also plots the output outlined in the multiple spectrogram plots with true labels. Using the AIS csv files, you can select a time period you are interested in, then generate a prediction and plot associated with that time period. You can play around with the range function and methods to filter the ais tracks if it is plotting too many/too few ais tracks. I would recommend considering a method to filter by mmsi as well.

![Uncertainty AIS plot](/example_plots/uncertainty_ais_example.png?raw=true "Uncertainty AIS Plot")


```
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

python3 main.py  --data_dir='/smallwork/beards/CS4321/acoustic_datasets/test_data/' \
                --checkpoint_dir='/smallwork/beards/CS4321/AVBW_Team/cs4321-team-sonar-final-project/thesis_working/plots/ais/' \
               --mode='plot' --plot_type='ais' --data_type='mel' --sample_rate=4000 \
               --channels=4 --start_date='20190505 084727' --end_date='20190505 145937'

date
```

checkpoint_dir is the plot output folder
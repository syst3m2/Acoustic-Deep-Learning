"""
    Andrew Pfau
    Sonar Classifier

    This program plots embeddings from the last layer of a model. That model must have been trained and saved as a checkpoint file.
    Options are PCA or TSNE

    This functions much like saved model where a test dataset is first loaded and inference is run on the saved model. The embeddings from
    the model are then extracted and plotted. The last layer of the model must be named 'last'. 

"""


import sys
import os
import argparse
import tensorflow as tf

from .data.dataset_processor import dataset_processor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def embeddings_plotter(hparams, input_gen):
    # for extracing embeddings for tSNE/ PCA
    # create dataset, batch size should not matter here
    data_proc = dataset_processor.AudioDataset(win_size=hparams.win_size, overlap=hparams.overlap, sample_points=hparams.sample_pts, mel_bins=hparams.mel_bins, dur=hparams.duration, modelType=hparams.model_type, 
                                            sample_rate=hparams.sample_rate, data_filepath=hparams.data_dir, epochs=1, batch_size=256, mode='test')

    test_data = data_proc.make_dataset(hparams.input_file, input_gen )

    model = tf.keras.models.load_model(hparams.checkpoint_path)
    embeddings_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('last').output)
    # embeddings is a 2D numpy array
    embeddings = embeddings_model.predict(test_data)

    x_data = []
    y_data = []
    if hparams.mode == 'TSNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=hparams.perplexity, n_iter=hparams.iterations)
        tsne_data = tsne.fit_transform(embeddings)
        # split x and y data
        x_data = tsne_data[:,0]
        y_data = tsne_data[:,1]

    elif hparams.mode == "PCA":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        x_data = pca_result[:,0]
        y_data = pca_result[:,1]

    true_labels = []
    for _, label in test_data.as_numpy_iterator():
        for l in label:
            true_labels.append(np.argmax(l))
    
    colors = {0:'red', 1:'blue', 2:'green', 3:'yellow' ,4:'orange', 5:'black', 6:'magenta', 7:'0.75'}
    name = dataset_processor.get_class_names()

    plt_colors = [colors[x] for x in true_labels]
    plt_labels = [name[x] for x in true_labels]

    # plot and save
    plt.figure(figsize=(10,10))
    if hparams.mode == 'TSNE':
        plt.scatter(x_data, y_data, c=plt_colors)

    if hparams.mode == 'PCA':
        data = pd.DataFrame([x_data, y_data])
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=data,
            legend="full",
            alpha=0.3
        )

    plt.savefig(hparams.output_file)
    plt.close()


def main():
    """
        Collect and process command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',          type=str,   help="Target directory with sample audio files")
    parser.add_argument('--input_file',        type=str,   help="Name of csv file listing sample inputs and labels. Must be in same location as data_dir. Default is test_labels.csv", default='test_labels.csv')
    parser.add_argument('--checkpoint_path',   type=str,   help="Path to checkpoint file")
    parser.add_argument('--mode',              type=str,   help="PCA or TNSE, default is PCA", default='PCA')
    parser.add_argument('--output_file',       type=str,   help="Name of output file. Will produce a png file")
    
    # spectrogram parameters
    parser.add_argument('--model_input',      type=str, default='stft', help="Input format into model, either stft or mfcc.")
    parser.add_argument('--win_size',         type=int, default=250,    help='Spectrogram window size in msec')
    parser.add_argument('--overlap',          type=int, default=50,     help='Spectrogram window overlap in percent')
    parser.add_argument('--sample_pts',       type=int, default=512,    help='Number of FFT sample points')
    parser.add_argument('--sample_rate',      type=int, default=4000,   help='Sample rate of input audio')
    parser.add_argument('--duration',         type=int, default=30,     help='Audio duration in seconds')
    parser.add_argument('--channels',         type=int, default=1,      help='Number of input channels, Default is 1')

    # TSNE args
    parser.add_argument('--perplexity',    type=int,   default=40,  help="TSNE perplexity, default is 40")
    parser.add_argument('--iterations',    type=int,   default=300, help="TSNE iterations, default is 300")

    args = parser.parse_args()
    
    # based on input type and number of channels, determine what generator type to use
    model_input_generator = 'stft'
    if args.model_input == 'stft' and args.channels == 4:
        model_input_generator = 'stft_multi_channel'
    elif args.model_input == 'mfcc' and args.channels == 4:
        model_input_generator = 'mfcc_multi_channel'
    elif args.model_input == 'mfcc' and args.channels == 1:
        model_input_generator = 'mfcc'

    if args.mode not in ['PCA', 'TSNE']:
        sys.exit("WARNING: Mode not supported, please try again.")

    embeddings_plotter(args, model_input_generator)    

if __name__ == "__main__":
    main()
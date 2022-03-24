"""
    Andrew Pfau
    Thesis Code

    This program converts higher sample rate files to lower sample rate, mostly used for conversion
    from 20kHz to 4kHz. Uses concurrent processes to perform conversion on many files faster. 
"""

import glob
import os
import librosa
import soundfile as sf
import concurrent.futures
import sys
import argparse

def downsample(filename, tgt_path, tgt_freq, orig_freq):
    base_filename, file_extension = os.path.split(filename)
    new_filename = tgt_path + file_extension
    signal, sr = librosa.load(filename, sr=orig_freq, mono=False)
    sig_4k = librosa.resample(signal, sr, tgt_freq)
    sf.write(new_filename, sig_4k, samplerate = tgt_freq, subtype='PCM_16')
    return new_filename

def main(file_dir, tgt_dir, original_sampleRate, target_sampleRate):
    # Create a pool of processes. By default, one is created for each CPU in your machine.
    max_cpu = os.cpu_count() - 1
    print("Using " + str(max_cpu) + " cpus")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpu) as executor:
        # Get a list of files to process
        audio_files = glob.glob(file_dir + "*.wav")
        num_files = len(audio_files)
       
        tgt_dir_list  = [tgt_dir]  * num_files
        orig_sr_list = [original_sampleRate] * num_files
        tgt_sr_list = [target_sampleRate]   * num_files
        # Process the list of files, but split the work across the process pool to use all CPUs!
        for audio_file, thumbnail_file in zip(audio_files, executor.map(downsample, audio_files, tgt_dir_list, tgt_sr_list, orig_sr_list)):
            print("A resampled file for " + str(audio_file) + " was saved as " + str(thumbnail_file))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_sr',   type=int, default=20000, help="Original sample rate of audio")
    parser.add_argument('--tgt_sr',    type=int, default=4000,  help="Target sample rate of audio to downsample to")
    parser.add_argument('--data_path', type=str, help="Path to save new downsampled files at")
    parser.add_argument('--tgt_path',  type=str, help="Path of target audio files")

    args = parser.parse_args()
    main(args.tgt_path, args.data_path, args.orig_sr, args.tgt_sr)
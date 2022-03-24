"""
	Andrew Pfau
	Thesis Work

	Program to store small scripts for random but repeated tasks like counting files in a folder based on label
	or deleting files in a folder that do not meet a cretian file size 
"""
# import libraries
import os
import glob
import pandas as pd
import argparse
import sys
import shutil
from sklearn.model_selection import train_test_split

def count_classes(tgt_dir, class_list):
	# count the number of each class in the tgt_dir based on the file's titles	
	# zero list of class counts
	class_counts = [0 for x in range(0,len(class_list))]
	for line in glob.glob(tgt_dir+ "*.wav"):
		for idx, label in enumerate(class_list):
			if label.strip() in line:
				class_counts[idx] += 1
	
	for label,count in zip(class_list, class_counts):
		print(label + ": " +str(count))
	total_samples = sum(class_counts)
	print("Total: " + str(total_samples))

#delete files that are not 30 seconds at 4k Hz and printout number of files removed
def del_small_files(tgt_dir, file_size):
	idx = 0
	removed = []
	for f in glob.glob(tgt_dir + '*.wav'):
		if os.path.getsize(f) < file_size:
			removed.append(f)
			os.remove(f)	
			idx +=1
	print(str(idx) + " files removed")
	data = pd.DataFrame(removed)
	data.to_csv("removed_files.csv", index=False, header=False)

def rename(tgt_dir):
	class_l = ['classA', 'classB', 'classC', 'classD', 'classE']
	for idx, f in enumerate(os.listdir(tgt_dir)):
		new_f_name = class_l[idx % 5] + '-' + str(round(idx*12.1, 2)) + '-' + str(idx *100.0) + '-_120101_'+str(idx) +'_test.wav'
		
		os.rename(os.path.join(tgt_dir, f), os.path.join(tgt_dir, new_f_name))

def main():
	"""
	Function to process command line arguments and create function path
	"""
	str_to_bool = lambda x : True if x.lower() == 'true' else False
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',        type=str,         default='phys',        help="The mode determines which function path to take. Options are: count-classes, delete")
	parser.add_argument('--data_dir',    type=str,         default='.',           help="Location of data files")
	parser.add_argument('--file_size',   type=int,         default=240044,       help="Size of files in bytes, if a file is less than this size it will be deleted. Use ls -la to find this value")

	parser.add_argument('--class_list',  type=str,         default='classA,classB,classC,classD,classE',    help="Comma seperated list of classes to count")

	args = parser.parse_args()

	if args.mode == 'count-classes':
		count_classes(args.data_dir, args.class_list.split(','))
	elif args.mode == 'delete':
		del_small_files(args.data_dir, args.file_size)
	elif args.mode == 'rename':
		rename(args.data_dir)
	else:
		sys.exit("WARNING: Mode not supported")

if __name__ == "__main__":
	main()

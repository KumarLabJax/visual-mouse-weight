import sys
import numpy as np

# Reads appended npy data
def read_data(filename):
	return_data = []
	with open(filename,'rb') as file:
		while True:
			try:
				if sys.version_info[0]==2:
					return_data.append(np.load(file))
				else:
					return_data.append(np.load(file, encoding = 'bytes', allow_pickle = False))
			except (IOError, ValueError):
				break
	return_data = np.reshape(return_data, [-1, np.shape(return_data)[-1]])
	# Overwrite because the data when appended at initial write-time is super slow to process.
	# This makes the second time you read in the data fast
	np.save(filename, return_data, allow_pickle=False)
	return return_data
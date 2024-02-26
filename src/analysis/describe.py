import pandas as pd
import numpy as np
import math, sys, os

# Mean: average value
def mean(x):
	return np.nansum(x) / x.shape[0]

# Median: middle value
def median(x):
	return x[x.shape[0] // 2] if x.shape[0] % 2 == 1 else (x[x.shape[0] // 2 - 1] + x[x.shape[0] // 2]) / 2

# Quartiles: 25%, 50%, 75%
def quartiles(x):
	return median(x[:x.shape[0] // 2 + (1 if x.shape[0] % 2 == 1 else 0)]), median(x), median(x[x.shape[0] // 2:])

# Standard deviation: measure of the amount of variation or dispersion of a set of values
def std(x):
	x = (x - mean(x)) ** 2
	return math.sqrt(np.nansum(x) / (x.shape[0]))

# Get description of a column
def get_description(col):
	col = col[~pd.isna(col)]
	x = np.sort(col)
	quarts = quartiles(x)
	return x.shape[0], np.round(mean(x), 2), np.round(std(x), 2), \
		np.round(quarts[0], 2), np.round(quarts[1], 2), np.round(quarts[2], 2), \
		np.round(x[0], 2), np.round(x[-1], 2)

def describe(dataset):
	return pd.DataFrame(data = np.apply_along_axis(get_description, 0, np.array(dataset)[:,6:]), 
		index = ["Count", "Mean", "Std", "25%", "50%", "75%", "Min", "Max"],
		columns = dataset.columns.values[6:])

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")
	print(describe(df))
	
	# Compare with the one of Pandas lib
	# print(df.iloc[:, 5:].describe())

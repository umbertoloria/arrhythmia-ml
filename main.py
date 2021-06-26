def get_lines(filepath):
	file = open(filepath, 'r')
	lines = list()
	for line in file.readlines():
		if line.endswith('\r\n'):
			line = line[:-2]
		if line.endswith('\r'):
			line = line[:-1]
		if line.endswith('\n'):
			line = line[:-1]
		lines.append(line)
	return lines


def load_dataset(data_filepath):
	import sys
	import pandas as pd
	data_lines = get_lines(data_filepath)
	records = list()
	for line in data_lines:
		# parts = line.split(',')
		# record = list()
		record = line.split(',')
		records.append(record)
	# width = len(records[0])
	# height = len(records)
	# print(width, height)
	columns = {
		'age': [],
		'sex': [],  # 0 <==> male, 1 <==> female
		'height': [],
		'weight': [],
		'qrs-duration': [],  # Average of QRS duration in milliseconds
		'pq-interval': [],  # Average duration between onset of P and Q waves in milliseconds
		'qt-interval': [],  # Average duration between onset of Q and offset of T waves in milliseconds
		't-interval': [],  # Average duration of T wave in milliseconds
		'p-interval': [],  # Average duration of P wave in milliseconds
		# Vector angles in degrees of front plane of:
		'qrs': [],
		't': [],
		'p': [],
		'qrst': [],
		'j': [],
	}
	mapping_cols_configs = {
		0: {'column': 'age', 'type': 'int'},
		1: {'column': 'sex', 'type': 'int'},
		2: {'column': 'height', 'type': 'int'},
		3: {'column': 'weight', 'type': 'int'},
		4: {'column': 'qrs-duration', 'type': 'int'},
		5: {'column': 'pq-interval', 'type': 'int'},
		6: {'column': 'qt-interval', 'type': 'int'},
		7: {'column': 't-interval', 'type': 'int'},
		8: {'column': 'p-interval', 'type': 'int'},
		9: {'column': 'qrs', 'type': 'int'},
		10: {'column': 't', 'type': 'int', 'optional': True},
		11: {'column': 'p', 'type': 'int', 'optional': True},
		12: {'column': 'qrst', 'type': 'int', 'optional': True},
		13: {'column': 'j', 'type': 'int', 'optional': True},
	}
	for record in records:
		for index, value in enumerate(record):
			if index in mapping_cols_configs:
				config = mapping_cols_configs[index]
				config_column = config['column']
				config_type = config['type']
				if config_type == 'int':
					try:
						value = int(value)
					except:
						if 'optional' in config.keys() and config['optional']:
							value = None
						else:
							print(f'ERROR: Int convertion of columns {config_column}', file=sys.stderr)
				elif config_type == 'float':
					try:
						value = float(value)
					except:
						if 'optional' in config.keys() and config['optional']:
							value = None
						else:
							print(f'ERROR: Float convertion of columns {config_column}', file=sys.stderr)
				columns[config_column].append(value)
	df = pd.DataFrame(data=columns)
	return df


def visualize_full(df):
	print(df.head())
	print(df.describe())
	import seaborn as sns
	plt = sns.pairplot(df[['age', 'height', 'weight']])
	plt.savefig('pairplot.png')


def main():
	data_filepath = 'dataset/arrhythmia.data'
	df = load_dataset(data_filepath)
	visualize_full(df)


# print(ds)


# print(f'Hi')


if __name__ == '__main__':
	main()

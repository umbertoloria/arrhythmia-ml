import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from dataset_load import load_dataset


def compose_points(col1, col2):
	if len(col1) != len(col2):
		raise Exception("Unable to build points: column sizes must be equal")
	it1 = iter(col1)
	it2 = iter(col2)
	points = list()
	for i in range(len(col1)):
		points.append([next(it1), next(it2)])
	return points


# K-means clustering
def compute_kmeans(df, n_clusters, xcol, ycol, filepath):
	kmeans_model = KMeans(n_clusters=n_clusters)
	points = compose_points(df[xcol].values, df[ycol].values)
	x = np.array(points)
	kmeans_result = kmeans_model.fit(x)
	# print(kmeans_result.cluster_centers_)
	cluster_column = f'cluster-{xcol}-{ycol}'
	df[cluster_column] = kmeans_result.labels_
	plt.figure()
	sns.scatterplot(x=xcol, y=ycol, data=df, hue=cluster_column, palette="Accent", legend=False)
	plt.savefig(filepath)


def main():
	data_filepath = 'dataset/arrhythmia.data'
	skip_records_with_empty_values = False
	dataset = load_dataset(data_filepath, skip_records_with_empty_values)

	df = pd.DataFrame(data=dataset)
	print(df.head())
	print(df.describe())

	# compute_kmeans(df, 2, 't', 'p', 'plots/scatterplot-t-p.png')
	compute_kmeans(df, 2, 'pq-interval', 'qt-interval', 'plots/scatterplot-pqinterval-qtinterval.png')
	compute_kmeans(df, 3, 'weight', 'qrs-duration', 'plots/scatterplot-weight-qrsduration.png')
	plot = sns.pairplot(df[['age', 'height', 'weight']])
	plot.savefig('plots/pairplot.png')


if __name__ == '__main__':
	main()

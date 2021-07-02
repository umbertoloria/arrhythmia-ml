import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids

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
	kmeans_model = KMeans(n_clusters=n_clusters)  # , init='k-means++'
	points = compose_points(df[xcol].values, df[ycol].values)
	x = np.array(points)
	kmeans_result = kmeans_model.fit(x)
	# print(kmeans_result.cluster_centers_)
	cluster_column = f'kmeans-clusters-{xcol}-{ycol}'
	df[cluster_column] = kmeans_result.labels_
	plt.figure()
	sns.scatterplot(x=xcol, y=ycol, data=df, hue=cluster_column, palette="Accent", legend=False)
	plt.savefig(filepath)


# K-menoids clustering
def compute_kmenoids(df, n_clusters, xcol, ycol, filepath):
	kmedoids_model = KMedoids(n_clusters=n_clusters)
	points = compose_points(df[xcol].values, df[ycol].values)
	x = np.array(points)
	kmenoids_result = kmedoids_model.fit(x)
	# print(kmenoids_result.cluster_centers_)
	cluster_column = f'kmenoids-clusters-{xcol}-{ycol}'
	df[cluster_column] = kmenoids_result.labels_
	plt.figure()
	sns.scatterplot(x=xcol, y=ycol, data=df, hue=cluster_column, palette="Accent", legend=False)
	plt.savefig(filepath)


# DBSCAN clustering
def compute_dbscan(df, eps, min_samples, xcol, ycol, filepath):
	dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
	points = compose_points(df[xcol].values, df[ycol].values)
	x = np.array(points)
	dbscan_result = dbscan_model.fit(x)
	# print(dbscan_result.cluster_centers_)
	cluster_column = f'kmeans-clusters-{xcol}-{ycol}'
	df[cluster_column] = dbscan_result.labels_
	plt.figure()
	sns.scatterplot(x=xcol, y=ycol, data=df, hue=cluster_column, palette="Accent", legend=False)
	plt.savefig(filepath)


def load_augmented_dataset(data_filepath, skip_records_with_empty_values):
	dataset = load_dataset(data_filepath, skip_records_with_empty_values)
	df_orig = pd.DataFrame(data=dataset)
	means = df_orig.mean()
	for col in ['t', 'p', 'qrst', 'j', 'heart-rate']:
		col_mean = round(means[col])
		dataset[col] = [(col_mean if val is None else val) for val in dataset[col]]
	return dataset


def main():
	data_filepath = 'dataset/arrhythmia.data'
	skip_records_with_empty_values = False
	dataset = load_augmented_dataset(data_filepath, skip_records_with_empty_values)
	df = pd.DataFrame(data=dataset)
	print(df.head())
	print(df.describe())

	# compute_kmeans(df, 2, 't', 'p', 'plots/scatterplot-t-p.png')
	compute_kmeans(df, 2, 'pq-interval', 'qt-interval', 'plots/kmeans-pqinterval-qtinterval.png')
	compute_kmenoids(df, 2, 'pq-interval', 'qt-interval', 'plots/kmenoids-pqinterval-qtinterval.png')
	compute_dbscan(df, 10, 0, 'pq-interval', 'qt-interval', 'plots/dbscan10-0-pqinterval-qtinterval.png')
	compute_dbscan(df, 10, 3, 'pq-interval', 'qt-interval', 'plots/dbscan10-3-pqinterval-qtinterval.png')
	compute_dbscan(df, 10, 6, 'pq-interval', 'qt-interval', 'plots/dbscan10-6-pqinterval-qtinterval.png')
	compute_dbscan(df, 10, 9, 'pq-interval', 'qt-interval', 'plots/dbscan10-9-pqinterval-qtinterval.png')

	compute_kmeans(df, 3, 'weight', 'qrs-duration', 'plots/kmeans-weight-qrsduration.png')
	compute_kmenoids(df, 2, 'weight', 'qrs-duration', 'plots/kmenoids-weight-qrsduration.png')

	plot = sns.pairplot(df[['age', 'height', 'weight']])
	plot.savefig('plots/pairplot.png')


if __name__ == '__main__':
	main()

import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5927)

class KMeans():
	def __init__(self, D, n_clusters):
		self.n_clusters = n_clusters
		self.cluster_centers = np.zeros((n_clusters, D))

	def init_clusters(self, data):
		### TODO
		### Initialize cluster_centers using n_clusters points sampled from data
		indices = np.random.choice(data.shape[0],self.n_clusters)
		self.cluster_centers = data[indices].reshape(self.cluster_centers.shape)
		### END TODO

	def pred(self, x):
		### TODO: Given a sample x, return id of closest cluster center
		return np.argmin(np.linalg.norm(self.cluster_centers - x,axis=1),axis=0)
		# c = [np.linalg.norm(self.cluster_centers[i,:] - x) for i in range(self.n_clusters)]
		# return np.argmin(c)
		### END TODO

	def train(self, data, max_iter=10000, epsilon=1e-4):
		for it in range(max_iter):
			### TODO
			### Declare and initialize required variables
			labels = []#num cluster 
			for i in range(self.n_clusters):
				labels.append([])
			new_centers = []
			### Update labels for each point
			for i in range(data.shape[0]):
				labels[self.pred(data[i,:])].append(data[i,:])
			### Update cluster centers
			### Note: If some cluster is empty, do not update the cluster center
			for idx,label_list in enumerate(labels):
				if not(label_list):
					new_centers.append(self.cluster_centers[idx])
				else:
					new_centers.append(np.sum(label_list,axis=0)/len(label_list))
			### Check for convergence
			### Stop if distance between each of the old and new cluster centers is less than epsilon
			if np.all(np.linalg.norm(new_centers - self.cluster_centers) < epsilon):
				break
			self.cluster_centers = np.array(new_centers).reshape(self.cluster_centers.shape)
			### END TODO
		return it

	def replace_by_center(self, data):
		out = np.zeros_like(data)
		for i, x in enumerate(data):
			out[i] = self.cluster_centers[self.pred(x)]
		return out

def experiment():
	for images in ['1', '2', '3']:
		image = plt.imread(f'data/{images}.png')
		x = image.reshape(-1, 3)
		for k in [2,5,10]:
			kmeans = KMeans(D=3, n_clusters=k)
			kmeans.init_clusters(x)
			kmeans.train(x)
			out = kmeans.replace_by_center(x)
			plt.figure()
			plt.title("clusters = {}".format(k))
			plt.imshow(out.reshape(image.shape))
			plt.savefig("{}_{}.png".format(images,k))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', default='1', choices=['1', '2', '3'])
	parser.add_argument('--k', default=5, type=int)

	args = parser.parse_args()

	image = plt.imread(f'data/{args.image}.png')
	x = image.reshape(-1, 3)
	kmeans = KMeans(D=3, n_clusters=args.k)
	kmeans.init_clusters(x)
	kmeans.train(x)
	out = kmeans.replace_by_center(x)
	plt.imshow(out.reshape(image.shape))
	plt.show()

	experiment()


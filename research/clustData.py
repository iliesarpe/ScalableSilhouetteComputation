import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
import math
import h5py
import json
import pandas as pd
from sklearn.cluster import KMeans
import kmedoids
from sklearn.metrics import pairwise_distances
import mkl


import random 

fin = sys.argv[1]
fr = open(fin, 'r')
params = json.loads(fr.readlines()[0])
fr.close()

dataname = params["dataset"]
outfile = params["outfile"]
kVec = [int(val) for val in params["kVals"]]
seeds = [int(val) for val in params["rSeed"]]
maxiters = int(params["clustIts"])
reCluster = bool(params["reCluster"])
huge = False
if "hugeReal" in params:
	huge =  params["hugeReal"]

kMedoidsMetrics = ["euclidean", "cosine", "manhattan", "sqeuclidean", "canberra"] 

#Read data
#dataset = pd.read_csv(dataname, sep=",", dtype=float).to_numpy().tolist()
dataset = pd.read_csv(dataname, sep=",", dtype=float, header=None).to_numpy()
#print(dataset.dtype)

fout = h5py.File(outfile, 'a')
#Write data objects
#Setup hierarchy in hdf5 file
if "clustering" not in fout:
	fileClustering = fout.create_group("clustering")
else:
	fileClustering = fout["clustering"]
	
baseClust = fileClustering.name
if baseClust + "/kmeans" not in fout:
	kmeansClusteringGroup = fileClustering.create_group("kmeans")
else:
	kmeansClusteringGroup = fout[baseClust + "/kmeans"]

#skip kmedoids on large data due to O(n^2) memory
if not huge:
	if baseClust + "/kmedoids" not in fout:
		kmedoidsClusteringGroup = fileClustering.create_group("kmedoids")
	else:
		kmedoidsClusteringGroup = fout[baseClust + "/kmedoids"]

	basekmed = kmedoidsClusteringGroup.name
	for metric in kMedoidsMetrics:
		if basekmed + "/" + metric not in fout:
			fileClustering.create_group(basekmed + "/" + metric)

print("recluster", reCluster)

#--------------------------KMEANS-----------------------
print("starting k-means")
for i, kVal in enumerate(kVec):
	print(kVal)
	random.seed(seeds[i])
	candidateGrp = kmeansClusteringGroup.name + "/k-"+str(kVal)
	if (candidateGrp in fout) and (not reCluster):
		continue

	kmeansRes =  KMeans(n_clusters=kVal, random_state=seeds[i], max_iter=maxiters, n_init=1).fit(dataset)

	if (candidateGrp in fout):
		del kmeansClusteringGroup["k-"+str(kVal)]
	kmeansKGroup = kmeansClusteringGroup.create_group("k-"+str(kVal))

	assignment = kmeansRes.labels_
	centers = kmeansRes.cluster_centers_
	#Dumping results
	kmeansKGroup.create_dataset("assignment", data=assignment, dtype="uint64")
	kmeansKGroup.create_dataset("centers", data=centers, dtype="float")


	#--------------------------KMEDOIDS-----------------------
print("starting k-medoids")
if not huge:
	for metr in kMedoidsMetrics:
		print(metr)
		candidateGrp = kmedoidsClusteringGroup.name + "/" + metr + "/k-"+str(kVec[0])
		if (candidateGrp in fout) and (not reCluster):
			print("Skipping as it exists", candidateGrp)
			continue

		diss = pairwise_distances(dataset, metric=metr) # (\sum_dimensions (p(i)-p(j))^2)^(1/2)
		for i, kVal in enumerate(kVec):
			print(kVal)
			medRes = kmedoids.fasterpam(diss, kVal, max_iter=maxiters)
			assignMed = medRes.labels
			medoids = medRes.medoids
			metricKGroup = kmedoidsClusteringGroup.create_group(metr + "/" + "k-"+str(kVal))
			metricKGroup.create_dataset("assignment", data=assignMed, dtype="uint64")
			metricKGroup.create_dataset("centers", data=medoids, dtype="uint64")

if "dataset" not in fout:
	fout.create_dataset("dataset", data=dataset, dtype="float")

fout.close()

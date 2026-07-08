import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import sys
import json
import subprocess
from multiprocessing import Pool
import h5py
from concurrent.futures import ThreadPoolExecutor

def remove_suffix(input_string, suffix):
	if suffix and input_string.endswith(suffix):
		return input_string[:-len(suffix)]
	return input_string

def dumpJson(data, filename):
	json_object = json.dumps(data)
	with open(filename, "w") as outfile:
		outfile.write(json_object)

LARGE_DATA = set(["bioKDD2004_preprocessed","loc-gowalla_totalCheckins_preprocessed","RNA_preprocessed","powerconsumption", "metro"])
USED_DATA = set(["bioKDD2004_preprocessed","loc-gowalla_totalCheckins_preprocessed","RNA_preprocessed","powerconsumption", "metro", "credit", "IOT", "shuttle_preprocessed", "breast"])
folderTmp = "tmp/"
scriptRandSph = "genSphere.py"
scriptRandCirc = "genCircle.py"
scriptRandGauss = "syntheticDataGeneration.py"
scriptClustData = "clustData.py"
scriptNoisy = "clustNoisy.py"
modeExSilh = 1
modeAppxSilh = 2
modeParallel = 3
CPPEXE = "../cppCode/efficientSilh" # executable
folderRealDataCsv = "data/real/" # folder where .hdf5, data is
folderRealDataProc = "data/real/clust/" # folder where to place clustered data
folderResReal = "results/real" # folder where to place results (single thread)
folderResParallel = "results/parallel" # folder where to place results (parallel)

mode = sys.argv[1]
# Generate the clustering configuration files (to be executed by clustData.py) and run the clustering
if mode == "clust-real":
	kvals = [2, 5, 10, 15, 20]
	seeds = [10,20,302102,23231,231234,6564,7857872,6562,742672,735783,756737,3773,673]
	iters = 50
	tasks = []
	recluster = False
	for i, fr in enumerate(os.listdir(folderRealDataCsv)):
		print(i, fr)
		if not fr.endswith(".csv"):
			continue
		huge = False
		if fr.split(".")[0] in LARGE_DATA:
			huge = True
		print(fr)
		outname = folderRealDataProc + remove_suffix(fr, ".csv") + ".hdf5"
		dict_conf = {
		"dataset":folderRealDataCsv+fr, 
		"outfile":outname, 
		"kVals":kvals, 
		"reCluster":recluster, 
		"hugeReal":huge, 
		"rSeed":seeds, 
		"clustIts":iters}

		jsonfilename = folderTmp + "confClustReal" + str(i)+".json"
		dumpJson(dict_conf, jsonfilename)
		task = ["python", scriptClustData, jsonfilename]
		tasks.append(task)
                #NOTE: disable the line below if you just want the configuration files
		subprocess.run(["python", scriptClustData, jsonfilename])
	print(tasks)


# Generate the clustering configuration files (to be executed by ./efficientSilh) and compute the exact silhouette values for all clustering configurations. Note that for large data, the exact values are not computed, but sampling is applied.
elif mode == "ex-silh-real":
	#kvals = [i for i in range(2,11)]
	tasks = []
	recompute = True
	clustMethod = ["kmeans", "kmedoids"]
	for i, fr in enumerate(os.listdir(folderRealDataProc)):
		if not fr.endswith(".hdf5"):
			continue
		try:
			huge = False
			if fr.split(".")[0] in LARGE_DATA:
				huge = True
				#continue
			print("processing", fr, huge)
			f5 = h5py.File(folderRealDataProc+fr, 'r')
			paths = []
			for cmet in clustMethod: 
				if cmet == "kmeans":
					if 'kmeans' in f5['clustering'].keys():
						for kval in f5['clustering/kmeans'].keys():
							print(kval)
							if ('clustering/kmeans/' + kval + "/exSilh" not in f5) or (recompute):
								paths.append('clustering/kmeans/' + kval)
				elif (not huge) and cmet == "kmedoids":
					if 'kmedoids' in f5['clustering'].keys():
						for kmeth in f5['clustering/kmedoids'].keys():
							for kval in f5['clustering/kmedoids/'+kmeth].keys():
								if ('clustering/kmedoids/' + kmeth + "/" + kval + "/exSilh" not in f5) or (recompute):
									paths.append('clustering/kmedoids/' + kmeth + "/" + kval)
			f5.close()
			outname = folderTmp + "tmp.txt"
			dict_conf = {
			"dataset":folderRealDataProc+fr, 
			"outfile":outname, 
			"small":not huge, 
			"mode": modeExSilh,
			"paths":paths}
			jsonfilename = folderTmp + "conf_clustReal" + str(i)+".json"
			dumpJson(dict_conf, jsonfilename)
			task = [CPPEXE, jsonfilename]
			#print(task)
			#continue
			tasks.append(task)
			subprocess.call([CPPEXE, jsonfilename])
		except:
			print("ERROR", fr)
			continue

# Generate the clustering configuration files (to be executed by ./efficientSilh) and run ALL approximate approaches for all various configurations. Note that some baselines will only run for small datasets 
elif mode == "approx-silh-real-ex":
	tValsLarge = [32, 64, 128]
	tValsTiny = [8, 16, 32, 64, 128]
	delta = 0.1
	iterations = 10
	perc = [0.01, 0.05, 0.1]
	threads_main = 1
	clustMethod = ["kmeans", "kmedoids"]
	tasks = []
	for i, fr in enumerate(os.listdir(folderRealDataProc)):
		if not fr.endswith(".hdf5"):
			continue
		try:
			print(i, fr)
			large = False
			if fr.split(".")[0] in LARGE_DATA:
				large = True
				#continue
			tVals = tValsLarge
			if not large:
				tVals = tValsTiny
			f5 = h5py.File(folderRealDataProc+fr, 'r')
			paths = []
			for cmet in clustMethod: 
				if cmet == "kmeans":
					if 'kmeans' in f5['clustering'].keys():
						for kval in f5['clustering/kmeans'].keys():
							paths.append('clustering/kmeans/' + kval)
				elif cmet == "kmedoids":
					if 'kmedoids' in f5['clustering'].keys():
						for kmeth in f5['clustering/kmedoids'].keys():
							for kval in f5['clustering/kmedoids/'+kmeth].keys():
								paths.append('clustering/kmedoids/' + kmeth + "/" + kval)
			f5.close()
			outname = folderResReal + "/" + fr.strip(".hdf5") + ".json"
			dict_conf = {
			"dataset":folderRealDataProc+fr, 
			"outname":outname, 
			"mode": modeAppxSilh,
			"paths":paths,
			"isSmall":not large,
			"threadsMain":threads_main,
			"tVals":tVals,
			"iters":iterations,
			"delta":delta,
			"percEst3":perc
			}
			jsonfilename = folderTmp + "conf_testRealEx" + str(i)+".json"
			dumpJson(dict_conf, jsonfilename)
			task = [CPPEXE, jsonfilename]
			tasks.append(task)
			subprocess.call([CPPEXE, jsonfilename])
		except:
			print("error on", fr)
			continue
	print(tasks)

# Generate the clustering configuration files (to be executed by ./efficientSilh) and run parallel experiments 
elif mode == "parallel-tests":
	tValsLarge = [512, 1024]
	delta = 0.1
	iterations = 5
	perc = [0.01]
	threads_main = 1
	thVec = [1,2,4,8,16]
	clustMethod = ["kmeans"]
	tasks = []
	for i, fr in enumerate(os.listdir(folderRealDataProc)):
		if not fr.endswith(".hdf5"):
			continue
		try:
			large = False
			if fr.split(".")[0] in LARGE_DATA:
				large = True
			if not large:
				continue
			tVals = tValsLarge
			f5 = h5py.File(folderRealDataProc+fr, 'r')
			paths = []
			for cmet in clustMethod: 
				if cmet == "kmeans":
					if 'kmeans' in f5['clustering'].keys():
						for kval in f5['clustering/kmeans'].keys():
							paths.append('clustering/kmeans/' + kval)
				elif cmet == "kmedoids":
					if 'kmedoids' in f5['clustering'].keys():
						for kmeth in f5['clustering/kmedoids'].keys():
							for kval in f5['clustering/kmedoids/'+kmeth].keys():
								paths.append('clustering/kmedoids/' + kmeth + "/" + kval)
			f5.close()
			outname = folderResParallel + "/" + fr.strip(".hdf5") + ".json"
			dict_conf = {
			"dataset":folderRealDataProc+fr, 
			"outname":outname, 
			"mode": modeParallel,
			"paths":paths,
			"isSmall":not large,
			"threadsMain":threads_main,
			"tVals":tVals,
			"iters":iterations,
			"delta":delta,
			"thPPS":thVec,
			"percEst3":perc
			}
			jsonfilename = folderTmp + "conf_testRealPar" + str(i)+".json"
			dumpJson(dict_conf, jsonfilename)
			task = [CPPEXE, jsonfilename]
			tasks.append(task)
			subprocess.call([CPPEXE, jsonfilename])
		except:
			print("error on", fr)
			continue
	print(tasks)

# Utility used to produce the table concerning dataset properties 
elif mode == "getDataStats":
	stats = []
	for i, fr in enumerate(os.listdir(folderRealDataProc)):
		if not fr.endswith(".hdf5"):
			continue
		try:
			if fr.split(".")[0] not in USED_DATA:
				continue
			f5 = h5py.File(folderRealDataProc+fr, 'r')
			n = f5['dataset'].shape[0]
			z = f5['dataset'].shape[1]
			mydata = {"dataset":fr, "points":n, "dim":z}
			stats.append(mydata)
		except:
			print("error on", fr)
			continue
	jsonfilename = "results/data-log.json"
	dumpJson(stats, jsonfilename)
	exit(0)

# Utility the configuration files used to cluster the datasets for the selection of the best value of $k$
elif mode == "clust-bestk":
	kvals = [i for i in range(2,16)]
	seeds = [10,20,302102,23231,231234,6564,7857872,6562,742672,735783,756737,3773,673, 394, 10291029, 84877]
	iters = 50
	tasks = []
	recluster = False
	question = "Data may exist already clustered, do you want to override?"
	prompt = " [y/n] "
	valid = {"y": True, "n": False}
	for i, fr in enumerate(os.listdir(folderRealDataCsv)):
		print(i, fr)
		if not fr.endswith(".csv"):
			continue
		huge = False
		if fr.split(".")[0] in LARGE_DATA:
			huge = True
		print(fr)
		outname = folderRealDataProc + remove_suffix(fr, ".csv") + ".hdf5"
		dict_conf = {
		"dataset":folderRealDataCsv+fr, 
		"outfile":outname, 
		"kVals":kvals, 
		"reCluster":True, 
		"hugeReal":True,  # avoid kmedoids
		"rSeed":seeds, 
		"clustIts":iters}

		jsonfilename = folderTmp + "confClustbestk" + str(i)+".json"
		dumpJson(dict_conf, jsonfilename)
		task = ["python", scriptClustData, jsonfilename]
		tasks.append(task)
	print(tasks)

# Compute the exact silhouette values for the best k selection
elif mode == "ex-silh-bestk":
	#kvals = [i for i in range(2,11)]
	tasks = []
	recompute = True
	clustMethod = ["kmeans"]
	for i, fr in enumerate(os.listdir(folderRealDataProc)):
		if not fr.endswith(".hdf5"):
			continue
		try:
			huge = False
			if fr.split(".")[0] in LARGE_DATA:
				huge = True
				#continue
			print("processing", fr, huge)
			f5 = h5py.File(folderRealDataProc+fr, 'r')
			paths = []
			if 'kmeans' in f5['clustering'].keys():
				for kval in f5['clustering/kmeans'].keys():
					print(kval)
					myk = float(kval.split("-")[-1])
					if ('clustering/kmeans/' + kval + "/exSilh" not in f5) or (recompute) and myk < 16:
						paths.append('clustering/kmeans/' + kval)
			f5.close()
			outname = folderTmp + "tmp.txt"
			dict_conf = {
			"dataset":folderRealDataProc+fr, 
			"outfile":outname, 
			"small":not huge, 
			"mode": modeExSilh,
			"paths":paths}
			jsonfilename = folderTmp + "conf_clustRealBestk" + str(i)+".json"
			dumpJson(dict_conf, jsonfilename)
			task = [CPPEXE, jsonfilename]
			#print(task)
			#continue
			tasks.append(task)
			subprocess.call([CPPEXE, jsonfilename])
		except:
			print("ERROR", fr)
			continue

# Compute the approximate silhouette for each k, over the selection of best k
elif mode == "approx-silh-bestk":
	tValsLarge = [10]
	tValsTiny = [10]
	delta = 0.01
	iterations = 200
	perc = [0.01, 0.05, 0.1]
	threads_main = 10
	clustMethod = ["kmeans"]
	tasks = []
	for i, fr in enumerate(os.listdir(folderRealDataProc)):
		if not fr.endswith(".hdf5"):
			continue
		try:
			print(i, fr)
			large = False
			if fr.split(".")[0] in LARGE_DATA:
				large = True
				#continue
			tVals = tValsLarge
			if not large:
				tVals = tValsTiny
			f5 = h5py.File(folderRealDataProc+fr, 'r')
			paths = []
			if 'kmeans' in f5['clustering'].keys():
				for kval in f5['clustering/kmeans'].keys():
					myk = int(kval.split("-")[-1])
					if myk < 16:
						paths.append('clustering/kmeans/' + kval)
			f5.close()
			outname = folderResReal + "/" + fr.strip(".hdf5") + "_k.json"
			dict_conf = {
			"dataset":folderRealDataProc+fr, 
			"outname":outname, 
			"mode": modeAppxSilh,
			"paths":paths,
			"isSmall":False, #avoid running baselines heavy or inprecise
			"threadsMain":threads_main,
			"tVals":tVals,
			"iters":iterations,
			"delta":delta,
			"percEst3":perc
			}
			jsonfilename = folderTmp + "conf_testRealBestk" + str(i)+".json"
			dumpJson(dict_conf, jsonfilename)
			task = [CPPEXE, jsonfilename]
			tasks.append(task)
			subprocess.call([CPPEXE, jsonfilename])
		except:
			print("error on", fr)
			continue
	print(tasks)

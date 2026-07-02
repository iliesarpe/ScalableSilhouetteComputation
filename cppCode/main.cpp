#include "structures.h"
#include "algorithms.h"
#include <iostream>
#include <ostream>
#include <functional>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <highfive/H5Easy.hpp>

using json = nlohmann::json;
using Structures::AlgorithmParameters;
#define KMEANS 1
#define SMALLFULL 2000
#define STANDALONE 4 // general-purpose run: bypass the experiment HDF5 clustering schema entirely

namespace
{
	// True if `path` ends with `ext` (case-sensitive), e.g. HasExtension("data.hdf5", ".hdf5") == true.
	bool HasExtension(const std::string& path, const std::string& ext)
	{
		return path.size() >= ext.size() &&
		       path.compare(path.size() - ext.size(), ext.size(), ext) == 0;
	}

	// Maps the "distance" field of a standalone-mode config to the EUCLID/SQUARED/... constants in algorithms.h.
	int DistTypeFromString(const std::string& name)
	{
		if(name == "euclidean") return EUCLID;
		if(name == "sqeuclidean") return SQUARED;
		if(name == "manhattan") return MANHATTAN;
		if(name == "cosine") return COSINE;
		if(name == "canberra") return CANBERRA;
		std::cerr << "WARNING: unknown distance \"" << name << "\", defaulting to euclidean" << std::endl;
		return EUCLID;
	}
}

int main(int argc, char *argv[])
{
	std::string jsonfile(argv[1]);
	std::ifstream f(jsonfile);
	json exe_spec = json::parse(f);
	f.close();

	//double epsilon = 0.05;
	std::string fpoints = exe_spec["dataset"].get<std::string>();
	std::cout << "Dataset is: " << fpoints << std::endl;
	//std::string fout = exe_spec["outfile"].get<std::string>();
	int dataTocluster = exe_spec["mode"].get<int>();
   if (dataTocluster == STANDALONE) // RUN AS LIBRARY (General purpose usage), > 1 reproduce a setting of the paper 
   {
      // ---- Load points: plain CSV (no header) or a flat HDF5 "dataset" array ----
		std::vector<Structures::Point<double>> pointset;
		AlgorithmParameters params;
		bool pointsFromHDF5 = HasExtension(fpoints, ".h5") || HasExtension(fpoints, ".hdf5");

		std::unique_ptr<H5Easy::File> hdf5File; // kept alive only if the input is HDF5
		if(pointsFromHDF5)
		{
			hdf5File = std::make_unique<H5Easy::File>(fpoints, H5Easy::File::ReadOnly);
			pointset = Structures::BuildDatasetRaw(*hdf5File, params); // reads the flat "/dataset" array
		}
		else
		{
			pointset = Structures::BuildDatasetRaw(fpoints); // plain CSV, one point per line
		}
		long long int n = static_cast<long long int>(pointset.size());
		if(n == 0)
		{
			std::cerr << "ERROR: no points were loaded from " << fpoints << std::endl;
			return 1;
		}
		std::cout << "Loaded " << n << " points from " << fpoints << std::endl;

		// ---- Load the cluster assignment: CSV (one cluster id per line) or a flat HDF5 dataset ----
		std::string assignPath = exe_spec["assignment"].get<std::string>();
		std::vector<int> labels;
		if(pointsFromHDF5 && exe_spec.value("assignmentInHDF5", true))
			labels = Structures::LoadLabelsHDF5(*hdf5File, assignPath); // assignPath is an in-file dataset path, e.g. "/assignment"
		else
			labels = Structures::LoadLabelsCSV(assignPath); // assignPath is a file on disk

		if(static_cast<long long int>(labels.size()) != n)
		{
			std::cerr << "ERROR: assignment has " << labels.size() << " entries but the dataset has "
			           << n << " points." << std::endl;
			return 1;
		}

		int k = exe_spec.value("k", -1);
		if(k <= 0)
			k = 1 + *std::max_element(labels.begin(), labels.end()); // infer k from labels if not given
		params.k = k;
		params.clusterAssignments = Structures::BuildClusterAssignments(labels, k);

		// ---- Algorithm parameters ----
		std::string distName = exe_spec.value("distance", std::string("euclidean"));
		params.distType = DistTypeFromString(distName);
		params.delta = exe_spec.value("delta", 0.01);
		params.threads = exe_spec.value("threads", 1);
		params.seed = exe_spec.value("seed", 100);
		params.sampleSizeDouble = {}; // subsampled global estimator (\hat{s}_3) is opt-in, see "subsampleSizes" below
		if(exe_spec.contains("subsampleSizes"))
			params.sampleSizeDouble = exe_spec["subsampleSizes"].get<std::vector<long long int>>();

		if(exe_spec.contains("t"))
		{
			params.t = exe_spec["t"].get<int>();
		}
		else
		{
			// Derive t from (epsilon, delta) using the sample-complexity bound of Lemma 1 (c=18):
			// t = ceil( (3c/eps^2) * ln(5nk/delta) )
			double epsilon = exe_spec.value("epsilon", 0.1);
			double c = 18.0;
			params.t = static_cast<int>(std::ceil((3.0 * c / (epsilon * epsilon)) * std::log((5.0 * n * k) / params.delta)));
			std::cout << "No \"t\" provided, derived t=" << params.t << " from epsilon=" << epsilon << std::endl;
		}

		std::cout << "Running silh-pps-all: n=" << n << " k=" << k << " t=" << params.t
		           << " distance=" << distName << " threads=" << params.threads << std::endl;

		AlgorithmResult res;
		auto tstart = std::chrono::high_resolution_clock::now();
		Algorithms::ApproximateSilhPPS(pointset, params, res);
		auto tend = std::chrono::high_resolution_clock::now();
		double wallSeconds = std::chrono::duration<double>(tend - tstart).count();

		// ---- Dump results as JSON: global estimate + per-point local estimates ----
		std::string outPath = exe_spec.value("outfile", std::string("silhouette_result.json"));
		json out = {
			{"n", n},
			{"k", k},
			{"t", params.t},
			{"delta", params.delta},
			{"distance", distName},
			{"threads", params.threads},
			{"runtime_seconds", wallSeconds},
			{"global_silhouette", res.approxSilh},
			{"local_silhouette", res.silhApproxValues}
		};
		std::ofstream outj(outPath);
		outj << out.dump(2) << '\n';
		outj.close();

		std::cout << "Done in " << wallSeconds << "s. Global silhouette estimate ~= " << res.approxSilh
		           << ". Results written to " << outPath << std::endl;
   }
   else if(dataTocluster == 1) // Compute Exact Silh
	{
		H5Easy::DumpOptions options{H5Easy::Compression()};
		H5Easy::File hdf5File(fpoints, H5Easy::File::ReadWrite);

		AlgorithmParameters params;
		std::vector<Structures::Point<double>> pointset = Structures::BuildDatasetRaw(std::ref(hdf5File), std::ref(params));
		std::vector<std::string> allPaths = exe_spec["paths"].get<std::vector<std::string>>();
		bool small = exe_spec["small"].get<int>();
		std::cout << "Dataset is: " << fpoints << std::endl;


		for(auto& currpath : allPaths)
		{
			std::string pathCopy = currpath;
			std::string delimiter = "/";
			size_t pos = 0;
			std::string token;
			std::vector<std::string> tokens;
			while ((pos = pathCopy.find(delimiter)) != std::string::npos) {
				token = pathCopy.substr(0, pos);
				std::cout << token << std::endl;
				pathCopy.erase(0, pos + delimiter.length());
				tokens.push_back(token);
			}
			int dist = -1;
			tokens.push_back(pathCopy);
			if(tokens[1] == "kmeans")
			{
				params.distType = EUCLID;
				dist = EUCLID;
			}
			else if (tokens[1] == "kmedoids")
			{
				if(tokens[2] == "cosine")
				{
					params.distType = COSINE;
					dist =COSINE;
				}
				else if(tokens[2] == "euclidean")
				{
					params.distType = EUCLID;
					dist =EUCLID;
				}
				else if(tokens[2] == "manhattan")
				{
					params.distType = MANHATTAN;
					dist =MANHATTAN;
				}
				else if(tokens[2] == "sqeuclidean")
				{
					params.distType = SQUARED;
					dist = SQUARED;
				}
				else if(tokens[2] == "canberra")
				{
					params.distType = CANBERRA;
					dist = CANBERRA;
				}
				
			}
			std::string kval = tokens.back().substr(tokens.back().find("-")+1, tokens.back().length()); // token is "scott"
			int k = stoi(kval);
			params.k = k;

			std::cout << "Processing: " << currpath << " k is: " << k << " dist is: " << dist << std::endl;
			Structures::LoadClustering(std::ref(hdf5File), std::ref(params), currpath, std::ref(pointset), tokens[1]);
			//Process k-means
			std::cout << "data has: " << pointset.size() << std::endl;
			long long int totN = 0;
			for(int i=0; i < params.clusterAssignments.size(); i++)
			{
				for(auto& pointIdx : params.clusterAssignments[i])
				{
					pointset[pointIdx].setClust(i);
				}
			}
			std::vector<double> exSilh(pointset.size(), 0);
    			auto tStart = std::chrono::steady_clock::now();
			auto tFinish = std::chrono::steady_clock::now();
			if(small) // Can afford exact computation
			{
    			tStart = std::chrono::steady_clock::now();
				exSilh = Algorithms::ComputeExactSilhouettePoints(pointset, params.k, dist);
				tFinish = std::chrono::steady_clock::now();
			}
			else // Average for each point uni and pps for iters times 
			{
				std::cout << "Running approx-ex" << std::endl;
				auto iParams = params;
				iParams.t = 800;
				long long int baseSeed = 45134;
				double delta = 0.01;
				iParams.delta = 0.1;
				std::vector<long long int> ss3(10); // placeholder
				iParams.sampleSizeDouble = ss3;
				//int iters = 2;
				int iters = 4;
				for(int j=0; j< iters; j++)
				{
					long long int seed = baseSeed * (j) + 10;
					iParams.seed = seed;
					AlgorithmResult resPPS;
					Algorithms::ApproximateSilhPPS(std::ref(pointset), std::ref(iParams), std::ref(resPPS));
					std::transform(exSilh.begin(), exSilh.end(), resPPS.silhApproxValues.begin(), exSilh.begin(),std::plus<double>());
					AlgorithmResult resUNI;
					Algorithms::ApproximateSilhUniform(std::ref(pointset), std::ref(iParams), std::ref(resUNI));
					std::transform(exSilh.begin(), exSilh.end(), resUNI.silhApproxValues.begin(), exSilh.begin(),std::plus<double>());
				}
				std::for_each(exSilh.begin(), exSilh.end(), [exSilh, iters](double& entry) {entry /= (2*iters);});
			}
			for(int i=0; i < exSilh.size(); i++)
			{
				if((exSilh[i] > 1) || (exSilh[i] < -1))
				{
					std::cout << i << " " << exSilh[i] << " " << currpath << '\n';
					exit(0);
				}
			}

			H5Easy::DumpOptions optionsEx{H5Easy::Compression(), H5Easy::DumpMode::Overwrite};
			std::chrono::duration<double> totRT1 = tFinish-tStart;
			double exrt = totRT1.count();
			std::string pathWrite = currpath + "/exSilh";
			H5Easy::dump(hdf5File, pathWrite, exSilh, optionsEx);
			H5Easy::dumpAttribute(hdf5File, pathWrite, "exrt", exrt, optionsEx);
		}
	}
	else if(dataTocluster == 2) // Run Appx Methods
	{
		H5Easy::DumpOptions options{H5Easy::Compression()};
		H5Easy::File hdf5File(fpoints, H5Easy::File::ReadWrite);

		AlgorithmParameters params;
		std::vector<Structures::Point<double>> pointset = Structures::BuildDatasetRaw(std::ref(hdf5File), std::ref(params));
		std::vector<std::string> allPaths = exe_spec["paths"].get<std::vector<std::string>>();
		std::vector<int> tVals = exe_spec["tVals"].get<std::vector<int>>();
		int iters = exe_spec["iters"].get<int>();
		std::string outJName = exe_spec["outname"].get<std::string>();
		double delta = exe_spec["delta"].get<double>();
		int thMain = exe_spec["threadsMain"].get<int>();
		bool small = exe_spec["isSmall"].get<int>();
		bool parselast = true;
		std::cout << "Threads to be used: " << thMain << std::endl;
		params.delta = delta;
		//double sampleSizeDouble = exe_spec["probDoubleSample"].get<double>();
		//int sSizeEst3 = exe_spec["est3Size"].get<int>();
		//double percEst3 =exe_spec["percEst3"].get<double>();
		std::vector<double> percEst3 =exe_spec["percEst3"].get<std::vector<double>>();
		std::vector<long long int> ss3;
		for(auto perc : percEst3)
		{
			long long int sSizeEst3 = static_cast<long long int>(std::ceil(pointset.size() * perc));
			ss3.push_back(sSizeEst3);
		}
		double bucketSize = 0.05;
		long long int baseSeed = 134;
		params.sampleSizeDouble = ss3;



		std::cout << "Comparing appx methods: " << '\n';
		std::vector<json> allres;
		for(auto& currpath : allPaths)
		{
			std::string pathCopy = currpath;
			std::string delimiter = "/";
			size_t pos = 0;
			std::string token;
			std::vector<std::string> tokens;
			while ((pos = pathCopy.find(delimiter)) != std::string::npos) {
				token = pathCopy.substr(0, pos);
				std::cout << token << std::endl;
				pathCopy.erase(0, pos + delimiter.length());
				tokens.push_back(token);
			}
			int dist = -1;
			tokens.push_back(pathCopy);
			if(tokens[1] == "kmeans")
			{
				params.distType = EUCLID;
				dist = EUCLID;
			}
			else if (tokens[1] == "kmedoids")
			{
				if(tokens[2] == "cosine")
				{
					params.distType = COSINE;
					dist =COSINE;
				}
				else if(tokens[2] == "euclidean")
				{
					params.distType = EUCLID;
					dist =EUCLID;
				}
				else if(tokens[2] == "manhattan")
				{
					params.distType = MANHATTAN;
					dist =MANHATTAN;
				}
				else if(tokens[2] == "sqeuclidean")
				{
					params.distType = SQUARED;
					dist = SQUARED;
				}
				else if(tokens[2] == "canberra")
				{
					params.distType = CANBERRA;
					dist = CANBERRA;
				}
			}
			int k =0;
			if(parselast)
			{
				std::string kval = tokens.back().substr(tokens.back().find("-")+1, tokens.back().length()); // token is "scott"
				k = stoi(kval);
			}
			else
			{
				std::string kval = tokens[2].substr(tokens[2].find("-")+1, tokens[2].length()); // token is "scott"
				k = stoi(kval);
			}
			params.k = k;

			std::cout << "Processing: " << currpath << " k is: " << k << " dist is: " << dist << '\n';
			Structures::LoadClustering(std::ref(hdf5File), std::ref(params), currpath, std::ref(pointset), tokens[1]);

			//Process k-means
			std::cout << "data has: " << pointset.size() << std::endl;
			long long int totN = 0;
			for(int i=0; i < params.clusterAssignments.size(); i++)
			{
				for(auto& pointIdx : params.clusterAssignments[i])
				{
					pointset[pointIdx].setClust(i);
				}
			}
			double exrt = -1;
			std::string exPath = currpath + "/exSilh";
			if(small)
			{
				try {
					exrt = H5Easy::loadAttribute<double>(hdf5File, exPath, "exrt");
				}
				catch (int errorCode) {
					exrt=-1;
				} 
			}
			std::vector<double> exactSilhVals = H5Easy::load<std::vector<double>>(hdf5File, exPath);
			std::cout << "SILH EX LOADEED: " << exactSilhVals.size() << " rt: " << exrt << std::endl;

			int totBuckets = static_cast<int>(std::ceil(2.0/bucketSize));
			std::cout << "SILH BUCKETS COMPUTING: " << totBuckets << std::endl;

			std::vector<int> pointToBucket(pointset.size(), -1);
			std::vector<double> bucketSizes(totBuckets, 0);
			std::vector<double> avgBucketSilh(totBuckets, 0);
			double globalSilh = 0;

			std::cout << "distance is: " << dist << " exact value is: " << exactSilhVals[0] << std::endl;
			std::cout << currpath << std::endl;

			for(int pp=0; pp< exactSilhVals.size(); pp++)
			{
				double val = exactSilhVals[pp];
				int myBucketVal = static_cast<int>(std::floor((1+val)/bucketSize));
				if(myBucketVal >= totBuckets)
					myBucketVal = totBuckets-1;
				if(myBucketVal < 0)
					myBucketVal = 0;
				pointToBucket[pp] = myBucketVal;
				bucketSizes[myBucketVal]++;
				avgBucketSilh[myBucketVal] += val;
				globalSilh += val;
			}
			
			globalSilh /= pointset.size();
			for(int v=0; v < avgBucketSilh.size(); v++)
			{
				if(bucketSizes[v] > 0)
					avgBucketSilh[v] /= bucketSizes[v];
			}
			std::cout << "SILH EX COMPUTED: "<< globalSilh << std::endl;


			// Deterministic
			std::vector<double> solSimpl(iters);
			std::vector<double> solFrah(iters);
			std::vector<double> solFastSquared(iters);
			std::vector<double> rtSimpl(iters);
			std::vector<double> rtFrah(iters);
			std::vector<double> rtFastSquared(iters);

			std::cout << "running appx baselines: " << std::endl;
			if(small)
			{
				#pragma omp parallel for num_threads(thMain)
				for(int it=0; it < iters; it++)
				{
					AlgorithmResult resSimpl;
					double res = Algorithms::SimplifiedSilhouette(std::ref(pointset), std::ref(params), std::ref(resSimpl));
					#pragma omp critical
					solSimpl[it] = resSimpl.approxSilh;
					#pragma omp critical
					rtSimpl[it] = resSimpl.totRTEst1;
					//std::cout << "Simpl: " << res << '\n';

					AlgorithmResult resFrah;
					res = Algorithms::FrahlingSolhler(std::ref(pointset), std::ref(params), std::ref(resFrah));
					#pragma omp critical
					solFrah[it] = resFrah.approxSilh;
					#pragma omp critical
					rtFrah[it] = resFrah.totRTEst1;
					/*
					if(dist == SQUARED)
					{
						AlgorithmResult resEuclid;
						res = Algorithms::FastSquaredEuclidSilhouette(std::ref(pointset), std::ref(params), std::ref(resEuclid));
						//std::cout << "Fast is: " << resEuclid.approxSilh << " while exact is: " << globalSilh << '\n';
						#pragma omp critical
						solFastSquared[it] = resEuclid.approxSilh;
						#pragma omp critical
						rtFastSquared[it] = resEuclid.totRTEst1;
					}
					*/
				}
			}
			json confDet = {
					{"solSimp", solSimpl},
					{"solFrah", solFrah},
					{"solFastSquared", solFastSquared},
					{"runtimeSimpl", rtSimpl},
					{"rtFrah", rtFrah}
				};
			//std::cout << "simpl: " << solSimpl[0] << " FS: " << solFrah[0] << std::endl;



			std::cout << "running pps and uni: " << std::endl;
			std::vector<json> confsT;
			long long int tIdx = 1;
			for(auto tv : tVals)
			{
				std::cout << "t is: " << tv << '\n';
				params.t = tv;

				std::vector<std::vector<double>> appxPPS(iters);
				std::vector<std::vector<double>> appxUNI(iters);
				// Sampling Based
				std::vector<double> solEst1(iters);
				std::vector<double> rtEst1(iters);

				std::vector<double> solEst2PPS(iters);
				std::vector<double> rtEst2PPS(iters);
				std::vector<double> solEst2UNI(iters);
				std::vector<double> rtEst2UNI(iters);

				//std::vector<double> solEst3PPS(iters);
				std::vector<std::vector<double>> rtEst3PPS(iters);
				std::vector<std::vector<double>> solEst3PPS(iters);
				std::vector<std::vector<double>> solEst3UNI(iters);
				std::vector<std::vector<double>> rtEst3UNI(iters);


				std::vector<std::vector<double>> errBucksPPS(iters);
				std::vector<std::vector<std::vector<double>>> errClustsPPS(iters);
				std::vector<std::vector<double>> errBucksUNI(iters);
				std::vector<std::vector<std::vector<double>>> errClustsUNI(iters);
				#pragma omp parallel for num_threads(thMain)
				for(int it=0; it < iters; it++)
				{
					auto innerParams = params;
					long long int seed = baseSeed * (it) + 10 + tIdx;
					innerParams.seed = seed;

					AlgorithmResult resEst1;
					Algorithms::ComputeEstimatorSubsample(std::ref(pointset), std::ref(innerParams), std::ref(resEst1));
					#pragma omp critical
					solEst1[it] = resEst1.approxSilh;
					#pragma omp critical
					rtEst1[it] = resEst1.totRTEst1;
					//std::cout << "est1: " << resEst1.approxSilh << " " << resEst1.totRTEst1 << std::endl;

					// PPS
					AlgorithmResult resPPS;
					Algorithms::ApproximateSilhPPS(std::ref(pointset), std::ref(innerParams), std::ref(resPPS));
					//std::cout << "distance is: " << params.distType << std::endl;
					if((pointset.size() < SMALLFULL) && (it==0))
					{
						#pragma omp critical
						appxPPS[it] = resPPS.silhApproxValues;
					}
					std::vector<std::vector<double>> errorsClusterPPS = Algorithms::ComputeErrorClusters(std::cref(exactSilhVals), std::cref(pointset), std::cref(resPPS.silhApproxValues), k);
					#pragma omp critical
					errClustsPPS[it] = errorsClusterPPS;
					std::vector<double> errBucketsPPS = Algorithms::ComputeErrorBuckets(std::cref(exactSilhVals), std::cref(resPPS.silhApproxValues), std::cref(pointToBucket), totBuckets);
					#pragma omp critical
					solEst2PPS[it] = resPPS.approxSilh;
					#pragma omp critical
					rtEst2PPS[it] = resPPS.totRTEst1;
					//std::cout << "est2: " << resPPS.approxSilh << " " << resPPS.totRTEst1 << std::endl;
					//exit(0);
					//solEst3PPS[it] = resPPS.doubleApproxSilh;
					#pragma omp critical
					solEst3PPS[it] = resPPS.multipleM;
					#pragma omp critical
					rtEst3PPS[it] = resPPS.totRTEst3;

					#pragma omp critical
					errBucksPPS[it] = errBucketsPPS;
					// UNIform

					AlgorithmResult resUNI;
					Algorithms::ApproximateSilhUniform(std::ref(pointset), std::ref(innerParams), std::ref(resUNI));
					if((pointset.size() < SMALLFULL) && (it==0))
					{
						#pragma omp critical
						appxUNI[it] = resUNI.silhApproxValues;
					}

					std::vector<std::vector<double>> errorsClusterUNI = Algorithms::ComputeErrorClusters(std::cref(exactSilhVals), std::cref(pointset), std::cref(resUNI.silhApproxValues), k);
					#pragma omp critical
					errClustsUNI[it] = errorsClusterUNI;
					std::vector<double> errBucketsUniform = Algorithms::ComputeErrorBuckets(std::cref(exactSilhVals), std::cref(resUNI.silhApproxValues), std::cref(pointToBucket), totBuckets);
					#pragma omp critical
					solEst2UNI[it] = resUNI.approxSilh;
					#pragma omp critical
					rtEst2UNI[it] = resUNI.totRTEst1;
					#pragma omp critical
					solEst3UNI[it] = resUNI.multipleM;
					#pragma omp critical
					rtEst3UNI[it] = resUNI.totRTEst3;

					#pragma omp critical
					errBucksUNI[it] = errBucketsUniform;
					//std::cout << "est1: " << solEst1[0] << " est2UNI " << solEst2UNI[0] << " est3UNI " << solEst3UNI[0][0] << '\n';
				}
					

				json conf = {
					{"t", tv},
					{"Est1Sol", solEst1},
					{"Est1RT", rtEst1},
					{"Est2SolPPS", solEst2PPS},
					{"Runtime2SolPPS", rtEst2PPS},
					{"Est2SolUNI", solEst2UNI},
					{"Runtime2SolUNI", rtEst2UNI},
					{"Est3SolPPS", solEst3PPS},
					{"Runtime3SolPPS", rtEst3PPS},
					{"Est3SolUNI", solEst3UNI},
					{"Runtime3SolUNI", rtEst3UNI},
					{"allPPS", appxPPS},
					{"allUNI", appxUNI},
					{"errClustsPPS", errClustsPPS},
					{"errClustsUNI", errClustsUNI},
					{"bucketsPPS", errBucksPPS},
					{"bucketsUNI", errBucksUNI}
				};
				confsT.push_back(conf);
				tIdx++;
			}
			std::vector<double> exdump = exactSilhVals;
			std::vector<std::vector<long long int>> assigns = params.clusterAssignments;
			if(!small || (fpoints=="data/real/clust/shuttle_preprocessed.hdf5")|| (fpoints=="data/real/clust/beans.hdf5")|| (fpoints=="data/real/clust/elnino.hdf5"))
			{
				exdump.clear();
				assigns.clear();
			}

			json confFin = {
					{"method", tokens[1]},
					{"distance", params.distType},
					{"k", params.k},
					{"tvect", tVals},
					{"percEst3", percEst3},
					{"globalSilh", globalSilh},
					{"silhBuckets", avgBucketSilh},
					{"bucketSizes", bucketSizes},
					{"pathClust", currpath},
					{"allEX", exdump},
					{"exRT", exrt},
					{"assignment", assigns},
					{"iterations", iters},
					{"samplingT", confsT},
					{"isSmall", small},
					{"resDet", confDet},
					{"delta", delta}
				};
			allres.push_back(confFin);
		}
		json out = {
			{"Dataset", fpoints},
			{"ALLRES", allres}
		};
		std::ofstream outj;
		outj.open(outJName);
		outj << out << '\n';
		outj.close();
	}
	else if(dataTocluster == 3) // Run PPS Parallel
	{
		H5Easy::DumpOptions options{H5Easy::Compression()};
		H5Easy::File hdf5File(fpoints, H5Easy::File::ReadWrite);

		AlgorithmParameters params;
		std::vector<Structures::Point<double>> pointset = Structures::BuildDatasetRaw(std::ref(hdf5File), std::ref(params));
		std::vector<std::string> allPaths = exe_spec["paths"].get<std::vector<std::string>>();
		std::vector<int> tVals = exe_spec["tVals"].get<std::vector<int>>();
		int iters = exe_spec["iters"].get<int>();
		std::string outJName = exe_spec["outname"].get<std::string>();
		double delta = exe_spec["delta"].get<double>();
		int thMain = exe_spec["threadsMain"].get<int>();
		bool small = exe_spec["isSmall"].get<int>();
		std::vector<int> thPPS = exe_spec["thPPS"].get<std::vector<int>>();
		std::cout << "Threads to be used: " << thMain << std::endl;
		params.delta = delta;
		std::vector<double> percEst3 =exe_spec["percEst3"].get<std::vector<double>>();
		std::vector<long long int> ss3;
		for(auto perc : percEst3)
		{
			long long int sSizeEst3 = static_cast<long long int>(std::ceil(pointset.size() * perc));
			ss3.push_back(sSizeEst3);
		}
		double bucketSize = 0.05;
		long long int baseSeed = 134;
		params.sampleSizeDouble = ss3;



		std::cout << "Comparing appx methods: " << '\n';
		std::vector<json> allres;
		for(auto& currpath : allPaths)
		{
			std::string pathCopy = currpath;
			std::string delimiter = "/";
			size_t pos = 0;
			std::string token;
			std::vector<std::string> tokens;
			while ((pos = pathCopy.find(delimiter)) != std::string::npos) {
				token = pathCopy.substr(0, pos);
				std::cout << token << std::endl;
				pathCopy.erase(0, pos + delimiter.length());
				tokens.push_back(token);
			}
			int dist = -1;
			tokens.push_back(pathCopy);
			if(tokens[1] == "kmeans")
			{
				params.distType = EUCLID;
				dist = EUCLID;
			}
			else if (tokens[1] == "kmedoids")
			{
				if(tokens[2] == "cosine")
				{
					params.distType = COSINE;
					dist =COSINE;
				}
				else if(tokens[2] == "euclidean")
				{
					params.distType = EUCLID;
					dist =EUCLID;
				}
				else if(tokens[2] == "manhattan")
				{
					params.distType = MANHATTAN;
					dist =MANHATTAN;
				}
				else if(tokens[2] == "sqeuclidean")
				{
					params.distType = SQUARED;
					dist = SQUARED;
				}
				else if(tokens[2] == "canberra")
				{
					params.distType = CANBERRA;
					dist = CANBERRA;
				}
			}

			std::string kval = tokens.back().substr(tokens.back().find("-")+1, tokens.back().length()); // token is "scott"
			int k = stoi(kval);
			params.k = k;

			std::cout << "Processing: " << currpath << " k is: " << k << " dist is: " << dist << '\n';
			Structures::LoadClustering(std::ref(hdf5File), std::ref(params), currpath, std::ref(pointset), tokens[1]);
			//Process k-means
			std::cout << "data has: " << pointset.size() << std::endl;
			long long int totN = 0;
			for(int i=0; i < params.clusterAssignments.size(); i++)
			{
				for(auto& pointIdx : params.clusterAssignments[i])
				{
					pointset[pointIdx].setClust(i);
				}
			}
			std::string exPath = currpath + "/exSilh";
			std::vector<double> exactSilhVals = H5Easy::load<std::vector<double>>(hdf5File, exPath);
			std::cout << "SILH EX LOADEED: " << exactSilhVals.size() << std::endl;

			int totBuckets = static_cast<int>(std::ceil(2.0/bucketSize));
			std::cout << "SILH BUCKETS COMPUTING: " << totBuckets << std::endl;

			std::vector<int> pointToBucket(pointset.size(), -1);
			std::vector<double> bucketSizes(totBuckets, 0);
			std::vector<double> avgBucketSilh(totBuckets, 0);
			double globalSilh = 0;

			std::cout << "distance is: " << dist << " exact value is: " << exactSilhVals[0] << '\n';
			std::cout << currpath << '\n';

			for(int pp=0; pp< exactSilhVals.size(); pp++)
			{
				double val = exactSilhVals[pp];
				int myBucketVal = static_cast<int>(std::floor((1+val)/bucketSize));
				//std::cout << "Tot bucks: " << totBuckets << " myval: " << myBucketVal << "My silh val: " << val << std::endl;
				if(myBucketVal == totBuckets)
					myBucketVal--;
				pointToBucket[pp] = myBucketVal;
				bucketSizes[myBucketVal]++;
				avgBucketSilh[myBucketVal] += val;
				globalSilh += val;
			}
			
			globalSilh /= pointset.size();
			for(int v=0; v < avgBucketSilh.size(); v++)
			{
				if(bucketSizes[v] > 0)
					avgBucketSilh[v] /= bucketSizes[v];
			}
			std::cout << "SILH EX COMPUTED: " << std::endl;


			std::cout << "running pps and uni: " << '\n';
			std::vector<json> confsT;
			long long int tIdx = 1;
			for(auto tv : tVals)
			{
				std::cout << "t is: " << tv << '\n';
				params.t = tv;

				std::vector<std::vector<double>> rts;
				std::vector<std::vector<double>> appxPPS(iters);

				std::vector<double> solEst2PPS(iters);
				std::vector<double> rtEst2PPS(iters);

				std::vector<std::vector<double>> rtEst3PPS(iters);
				std::vector<std::vector<double>> solEst3PPS(iters);

				std::vector<std::vector<double>> errBucksPPS(iters);
				std::vector<std::vector<std::vector<double>>> errClustsPPS(iters);
				for(auto thr : thPPS)
				{
					std::vector<double> rtT(iters, 0);
					for(int it=0; it < iters; it++)
					{
						//std::cout << "it is:" << it << std::endl;
						auto innerParams = params;
						innerParams.threads = thr;
						long long int seed = baseSeed * (it) + 10 + tIdx;
						innerParams.seed = seed;

						// PPS
						AlgorithmResult resPPS;
						Algorithms::ApproximateSilhPPS(std::ref(pointset), std::ref(innerParams), std::ref(resPPS));
						std::vector<std::vector<double>> errorsClusterPPS = Algorithms::ComputeErrorClusters(std::cref(exactSilhVals), std::cref(pointset), std::cref(resPPS.silhApproxValues), k);
						errClustsPPS[it] = errorsClusterPPS;
						std::vector<double> errBucketsPPS = Algorithms::ComputeErrorBuckets(std::cref(exactSilhVals), std::cref(resPPS.silhApproxValues), std::cref(pointToBucket), totBuckets);
						solEst2PPS[it] = resPPS.approxSilh;
						rtT[it] = resPPS.totRTEst1;
						rtEst2PPS[it] = resPPS.totRTEst1;
						solEst3PPS[it] = resPPS.multipleM;
						rtEst3PPS[it] = resPPS.totRTEst3;
						errBucksPPS[it] = errBucketsPPS;

					}
					rts.push_back(rtT);
				}
					

				json conf = {
					{"t", tv},
					{"threads", thPPS},
					{"runtimesPar", rts},
					{"Est3SolPPS", solEst3PPS},
					{"Runtime3SolPPS", rtEst3PPS},
					{"allPPS", appxPPS},
					{"errClustsPPS", errClustsPPS},
					{"bucketsPPS", errBucksPPS}
				};
				confsT.push_back(conf);
				tIdx++;
			}
			std::vector<double> exdump = exactSilhVals;
			std::vector<std::vector<long long int>> assigns = params.clusterAssignments;
			if(!small)
			{
				exdump.clear();
				assigns.clear();
			}

			json confFin = {
					{"method", tokens[1]},
					{"distance", params.distType},
					{"k", params.k},
					{"tvect", tVals},
					{"threads", thPPS},
					{"globalSilh", globalSilh},
					{"silhBuckets", avgBucketSilh},
					{"bucketSizes", bucketSizes},
					{"allEX", exdump},
					{"pathClust", currpath},
					{"assignment", assigns},
					{"iterations", iters},
					{"samplingT", confsT},
					{"isSmall", small},
					{"delta", delta}
			};
			allres.push_back(confFin);
		}
		json out = {
			{"Dataset", fpoints},
			{"ALLRES", allres}
		};
		std::ofstream outj;
		outj.open(outJName);
		outj << out << '\n';
		outj.close();
	}
    	return 0;
}

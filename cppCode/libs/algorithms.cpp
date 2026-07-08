#include "algorithms.h"

std::vector<long long int> Algorithms::sublinearBinomialSample(long long int trials, double probability, std::mt19937_64& generator)
{
    long long int position  = 0;
    std::vector<long long int> sampled;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double x = unif(generator);
    double c = std::log(1-probability);
    if(x < probability)
    {
        sampled.push_back(position);
    }
    while(position < trials)
    {
        x = unif(generator);
        double y = std::log(x)/c;
        position = position + std::floor(y) + 1;
        if(position < trials)
        {
            sampled.push_back(position);
        }
    }
    return sampled;
}

std::vector<double> Algorithms::ComputeErrorBuckets(const std::vector<double>& exSilh, const std::vector<double>& appxSilh, const std::vector<int>& pointBuckets, int totBucks)
{
	std::vector<double> supErrBuckets(totBucks, 0); 
	for(int i=0; i < pointBuckets.size(); i++)
	{
		double errorPoint = std::abs(exSilh[i] - appxSilh[i]);
		supErrBuckets[pointBuckets[i]] = std::max(errorPoint,supErrBuckets[pointBuckets[i]]);
		if(errorPoint > 2)
		{
			std::cout << "ERRORR: " <<  i << " " << exSilh[i] << " "  << errorPoint <<  " " <<appxSilh[i] << " " << std::endl;
			//exit(0);
		}

	}
	return supErrBuckets;
}

std::vector<std::vector<double>> Algorithms::ComputeErrorClusters(const std::vector<double>& exSilh, const std::vector<Point<double>>& pointset, const std::vector<double>& appxSilh,int k)
{
	std::vector<std::vector<double>> errors;
	for(int i=0; i< k; i++) // very lazy for readability
	{
		errors.push_back(std::vector<double>(2,0)); // For each cluster retain maximum and average errors over points, i.e. |s(e) - \hat{s}(e)| 
	}

	std::vector<long long int> sizes(k, 0);
	for(int i=0; i < pointset.size(); i++)
	{
		int myclust = pointset[i].getClust();
		sizes[myclust]++;
		double errorPoint = std::abs(exSilh[i] - appxSilh[i]);
		errors[myclust][0] = std::max(errorPoint, errors[myclust][0]);
		errors[myclust][1] += errorPoint;
		if(errorPoint > 2)
		{
			std::cout << "ERRORR: " <<  i << " " << exSilh[i] << " "  << errorPoint <<  " " <<appxSilh[i] << " " << std::endl;
			//exit(0);
		}
	}
	for(int i=0; i< k; i++) // very lazy for readability
	{
		if(sizes[i]>0)
			errors[i][1] /= sizes[i];
	}
	return errors;
}
   
std::vector<Point<double>> Algorithms::InitializeCentersKMeans(std::vector<Point<double>>& pointset, int k, bool kmeansPP, long long int seed)
{
    std::vector<Point<double>> centers;
    std::mt19937_64 gen64;
    gen64.seed(seed);
    // Random Initialization
    if(!kmeansPP)
    {
        std::uniform_int_distribution<long long int> randomPoint(0, pointset.size()-1); // random [0, n-1]
        std::unordered_set<long long int> previous_points;
        for(int i=0; i < k; i++)
        {
            long long int currentPointIdx = randomPoint(gen64);
            
            // Avoid selecting the same point multiple times, this happens with very tiny probability
            while(previous_points.find(currentPointIdx) != previous_points.end())
                currentPointIdx = randomPoint(gen64);
            
            // Now the index corresponds to a new center
            centers.push_back(pointset[currentPointIdx]);
        }
    }
    else // KMeans++
    {
        std::uniform_int_distribution<long long int> randomPoint(0, pointset.size()-1); // random [0, n-1]
        centers.push_back(pointset[randomPoint(gen64)]); 
        std::vector<double> distances(pointset.size(), 0);
        for(int i=0; i < pointset.size(); i++)
        {
            distances[i] = ComputeDistancePoints(pointset[i], centers[0], SQUARED);
            pointset[i].setClust(0);
        }
        for(int i=0; i<k-1; i++)
        {
            std::discrete_distribution<> weightDistrib(distances.begin(), distances.end());
            long long int nextIdx = weightDistrib(gen64);
            centers.push_back(pointset[nextIdx]); 
            for(int j=0; j < pointset.size(); j++)
            {
                double currentDistance = ComputeDistancePoints(pointset[j], pointset[nextIdx], SQUARED);
                if(distances[j] > currentDistance)
                {
                    distances[j] = currentDistance;
                    pointset[j].setClust(i);
                }
            }
        }

    }
    return centers; // Inefficient but O(kd), change to reference
}


double Algorithms::KMeans(std::vector<Point<double>>& pointset, std::vector<Point<double>>& centroids, int k, int iterations, bool kmeansPP, long long int seed)
{
	std::cout << "Initializing the centroids: " << std::endl;
    std::vector<Point<double>> centers = InitializeCentersKMeans(pointset, k, kmeansPP, seed);
	std::cout << "Centers have size: " << centers.size() << std::endl;
    long long int totDimensions = pointset[0].getDims();
    //Naive K-means Implementation
    for(int i=0; i < iterations; i++)
    {
        // Loop points to compute the distances from current centers O(nkd)
        for(int j=0; j<pointset.size(); j++)
        {
            double minDistance = ComputeDistancePoints(pointset[j], centers[0]);
            double minCenterIdx = 0;
            // Loop the remaining k-1 centroids
            for(int l=1; l < centers.size(); l++)
            {
                double currentDistance = ComputeDistancePoints(pointset[j], centers[l]);
                if(minDistance > currentDistance)
                {
                    minDistance = currentDistance;
                    minCenterIdx = l;
                }
            }
            pointset[j].setClust(minCenterIdx);
        }
        // Update centroids O(d(n+k))
        std::vector<Point<double>> updatedCentroids(k, Point<double>(totDimensions)); // Empty centroids
        std::vector<long long int> totPoints(k);
        for(int j=0; j<pointset.size(); j++)
        {
            totPoints[pointset[j].getClust()]++;
            for(int l=0; l<totDimensions; l++)
            {
                updatedCentroids[pointset[j].getClust()].updateCoord(l, pointset[j].getCoord(l));
            }
        }

        for(int j=0; j < k; j++)
        {
            if(totPoints[j] > 0)
            {
                for(int l=0; l<totDimensions; l++)
                {
                    updatedCentroids[j].updateCoord(l, totPoints[j], DIVIDE); // normalize the sum by the number of points assigned to the centroid
                }
            }
        }
        centers = updatedCentroids;
    }
	centroids = centers;

    return 0;
}

std::vector<double> Algorithms::ComputeExactSilhouettePoints(std::vector<Point<double>>& pointset, int k, int distance)
{
	std::vector<double> silhouettePoints(pointset.size(), 0);
	std::vector<long long int> clustSizes(k,0);
	for(auto& p : pointset) // O(n)
	{
		clustSizes[p.getClust()]++;
	}
		
	//#pragma omp parallel for num_threads(1)
	for(int i=0; i < pointset.size(); i++)
	{
		auto x = pointset[i];
		auto myClust = x.getClust();
		std::vector<double> averageDistances(k,0);

		for(int j=0; j < pointset.size(); j++)
		{
			if(i!=j)
			{
				averageDistances[pointset[j].getClust()] += ComputeDistancePoints(x, pointset[j], distance);
			}
		}
		double a = 0;
		double b = std::numeric_limits<double>::max();
		for(int j=0; j < k; j++)
		{
			if(j!= myClust)
			{
				if(clustSizes[j] > 0)
				{
					averageDistances[j] /= clustSizes[j];
					b = std::min(b, averageDistances[j]);
				}
			}
		}

		if(clustSizes[myClust] > 1)
			a = averageDistances[myClust]/(clustSizes[myClust]-1);
		else
			a = 0;

		double silh = (b-a)/(std::max(b, a));
		//#pragma omp critical
		silhouettePoints[i] = silh;
		//silhouettePoints.push_back(silh);
	}

	return silhouettePoints; // Copy constructor can be slow for high number N

}

void Algorithms::ApproximateSilhPPS(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res)
{
    //Cluster assignments is a vector of size k where each entry contains a vector contaning the indices of the points in the i-th cluter from pointset
    std::mt19937_64 gen;
    gen.seed(params.seed);
    std::uniform_real_distribution<double> uniform01(0.0,1.0); // Used to sample points in the final weighted sample
    long long int n = pointset.size();
    auto distanceType = params.distType;
    std::vector<std::vector<long long int>>& clusterAssignments = params.clusterAssignments; // Check if copy
    int k = clusterAssignments.size();
	double delta = params.delta;
	int threads = params.threads;	


    //std::vector<bool> tinyClusters(k, true);
    std::vector<std::vector<std::pair<long long int, double>>> weightedSamplePPS(k);

    auto tStart = std::chrono::steady_clock::now();
    for(int i=0; i < k; i++)
    {
		//std::cout << "Processing: " << i << "with size: " << clusterAssignments[i].size() << std::endl;
        if(params.t >= clusterAssignments[i].size()) // Current cluster has size at most t
        {
            std::vector<std::pair<long long int, double>> weightedSampleCluster;
            for(int j=0; j < clusterAssignments[i].size(); j++)
                    weightedSampleCluster.push_back(std::make_pair(clusterAssignments[i][j], 1.));
            weightedSamplePPS[i] = weightedSampleCluster;
            //clusterSamples[i] = clusterAssignments[i];
            //probabsSamples[i] = std::vector<double>(clusterAssignments[i].size(), 1.0);
			//std::cout << "ith cluster is small hence: " << i << " has weighted sample size: " << weightedSamplePPS[i].size() << std::endl;
        }
        else
        {
            //tinyClusters[i] = false;
            double samplingProb = 2.0/clusterAssignments[i].size() * std::log((2.*k)/delta);
			samplingProb = std::min(samplingProb, 1.0);
            std::vector<long long int> posFirstSample = sublinearBinomialSample(clusterAssignments[i].size(), samplingProb, std::ref(gen));
			//std::cout << "Size of first sample: " << posFirstSample.size() << std::endl;
            std::vector<double> weightsFirstSample(posFirstSample.size(), 0);
            //for(auto& pointIdx : posFirstSample) // O(log(k/delta) * |C_i|)
			#pragma omp parallel for num_threads(threads)
            for(int j=0; j< posFirstSample.size(); j++) // expected O(log(k/delta) * |C_i|)
            {
                auto& pointIdx = posFirstSample[j];
                double myweight = 0;
                //for(auto& pointIdx2 : clusterAssignments[i])
                for(int l=0; l< clusterAssignments[i].size(); l++)
                {
                    auto& pointIdx2 = clusterAssignments[i][l];
                    //myweight += distanceClusters(pointset[pointIdx], pointset[pointIdx2]);
                    myweight += ComputeDistancePoints(pointset[pointIdx], pointset[pointIdx2], distanceType);
                }
                //weightsFirstSample.push_back(myweight);
				#pragma omp critical
                weightsFirstSample[j] = myweight;
            }

            // Compute sampling probability of each point
            //for(auto& pointIdx : clusterAssignments[i])
            std::vector<std::pair<long long int, double>> weightedSampleCluster;
			#pragma omp parallel for num_threads(threads)
            for(int j=0; j < clusterAssignments[i].size(); j++)
            {
                auto& pointIdx = clusterAssignments[i][j];
                double maxWeight = 0;
                for(int idx=0; idx < posFirstSample.size(); idx++)
                {
                    auto pointIdx2 = posFirstSample[idx];
                    //double myweight = distanceClusters(pointset[pointIdx], pointset[pointIdx2]);
                    double myweight = ComputeDistancePoints(pointset[pointIdx], pointset[pointIdx2], distanceType);
                    if(weightsFirstSample[idx] > 0)
                    {
                        myweight /= weightsFirstSample[idx];
						#pragma omp critical
                        maxWeight = std::max(myweight, maxWeight);
                    }
                }
				#pragma omp critical
                maxWeight = std::max(maxWeight, 1./clusterAssignments[i].size());
				#pragma omp critical
                maxWeight = std::min(params.t * maxWeight, 1.);
                if(uniform01(gen) <= maxWeight) // sampling w.p. \gamma_e
                {
					#pragma omp critical
                    weightedSampleCluster.push_back(std::make_pair(pointIdx, maxWeight));
                }
            }
			#pragma omp critical
            weightedSamplePPS[i] = weightedSampleCluster;
			//std::cout << "ith cluster: " << i << " has weighted sample size: " << weightedSamplePPS[i].size() << std::endl;
        }
    }
    auto tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT1 = tFinish-tStart;
	res.stepART = totRT1.count();
	//std::cout << "Step 1 RT: " << res.stepART << " vs " << totRT1.count() << std::endl;

    tStart = std::chrono::steady_clock::now();
	double estimate = 0;

	if (params.globalOnly)
	{
		// Fast global path: evaluate only m randomly chosen points — O(m) instead of O(n).
		// When globalM <= 0 or >= n, fall back to evaluating all n points.
		long long int m = (params.globalM > 0 && params.globalM < n) ? params.globalM : n;
		std::uniform_int_distribution<long long int> distrib(0, n - 1);
		for (long long int i = 0; i < m; i++)
		{
			long long int pos = (params.globalM > 0 && params.globalM < n) ? distrib(gen) : i;
			auto vals = GetEstimateTerms(std::cref(weightedSamplePPS), std::cref(pointset),
			                             pointset[pos], std::cref(clusterAssignments), distanceType);
			double aHat = vals.first;
			double bHat = vals.second;
			estimate += (bHat - aHat) / std::max(bHat, aHat);
		}
		estimate /= m;
		auto tFinish2 = std::chrono::steady_clock::now();
		std::chrono::duration<double> totRT2 = tFinish2 - tStart;
		res.silhApproxValues.clear();
		res.approxSilh  = estimate;
		res.totRTEst1   = res.stepART + totRT2.count();
		return;
	}

	// Local path: evaluate all n points and return per-point silhouette values.
	std::vector<double> estimateSilh(n);
	#pragma omp parallel for num_threads(threads)
	for(int i=0; i < n; i++)
	{
		auto vals = GetEstimateTerms(std::cref(weightedSamplePPS), std::cref(pointset), pointset[i], std::cref(clusterAssignments), distanceType);
		double aHat = vals.first;
		double bHat = vals.second;
		double sHat = (bHat - aHat)/(std::max(bHat, aHat));
		#pragma omp critical
		estimateSilh[i] = sHat;
		#pragma omp critical
		estimate += sHat;
	}
	estimate /= n;
    tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT = tFinish-tStart;

	res.silhApproxValues.clear();
	res.silhApproxValues = estimateSilh;
	res.approxSilh = estimate;
	res.totRTEst1 = res.stepART + totRT.count();

	std::vector<long long int> samp3 = params.sampleSizeDouble;
	for(auto mSizeEst3 : samp3)
	{
		tStart = std::chrono::steady_clock::now();
		estimate = 0;
		std::uniform_int_distribution<> distrib(0, n-1);
		std::vector<long long int> posSelected;
		for(int i=0; i < mSizeEst3; i++)
		{
			posSelected.push_back(distrib(gen));
		}
		for(auto& pos : posSelected)
		{
			auto vals = GetEstimateTerms(std::cref(weightedSamplePPS), std::cref(pointset), pointset[pos], std::cref(clusterAssignments), distanceType);
			double aHat = vals.first;
			double bHat = vals.second;
			double sHat = (bHat - aHat)/(std::max(bHat, aHat));
			estimate += sHat;
		}
		estimate = estimate / mSizeEst3;
		tFinish = std::chrono::steady_clock::now();
		totRT = tFinish-tStart;
		res.multipleM.push_back(estimate);
		res.totRTEst3.push_back(res.stepART + totRT.count());
	}
	return;
	
}


void Algorithms::ApproximateSilhUniform(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res)
{
    //Cluster assignments is a vector of size k where each entry contains a vector contaning the indices of the points in the i-th cluter from pointset
    std::mt19937_64 gen;
    gen.seed(params.seed);

    std::uniform_real_distribution<double> uniform01(0.0,1.0); // Used to sample points in the final weighted sample
    long long int n = pointset.size();
    auto distanceType = params.distType;
    std::vector<std::vector<long long int>>& clusterAssignments = params.clusterAssignments; // Check if copy
    int k = clusterAssignments.size();

    std::vector<std::vector<std::pair<long long int, double>>> weightedSample(k);

    auto tStart = std::chrono::steady_clock::now();
    for(int i=0; i < k; i++)
    {
        if(params.t >= clusterAssignments[i].size()) // Current cluster has size at most t
        {
            std::vector<std::pair<long long int, double>> weightedSampleCluster;
            for(int j=0; j < clusterAssignments[i].size(); j++)
                    weightedSampleCluster.push_back(std::make_pair(clusterAssignments[i][j], 1.));
            weightedSample[i] = weightedSampleCluster;
        }
        else
        {
            double samplingProb = (1.*params.t)/clusterAssignments[i].size();
			//std::cout << "Sampling Prob is: " << samplingProb << '\n';
            std::vector<std::pair<long long int, double>> weightedSampleCluster;
            for(int j=0; j < clusterAssignments[i].size(); j++)
            {
                if(uniform01(gen) <= samplingProb)
                {
                    weightedSampleCluster.push_back(std::make_pair(clusterAssignments[i][j], samplingProb));
                }
            }
			if(weightedSampleCluster.size() == 0)
            	weightedSampleCluster.push_back(std::make_pair(clusterAssignments[i][0], samplingProb));
            weightedSample[i] = weightedSampleCluster;
        }
    }
    auto tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT1 = tFinish-tStart;
	res.stepART = totRT1.count();

    tStart = std::chrono::steady_clock::now();
    double estimate = 0;
	std::vector<double> estimateSilh(n);
	for(int i=0; i < n; i++)
	{
		auto vals = GetEstimateTerms(std::cref(weightedSample), std::cref(pointset), pointset[i], std::cref(clusterAssignments), distanceType);
		double aHat = vals.first;
		double bHat = vals.second;
		double sHat = (bHat - aHat)/(std::max(bHat, aHat));
		estimateSilh[i] = sHat;
		estimate += sHat;
	}
	estimate /= n;
    tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT = tFinish-tStart;

	res.silhApproxValues.clear();
	res.silhApproxValues = estimateSilh;
	res.approxSilh = estimate;
	res.totRTEst1 = res.stepART + totRT.count();
    

	//int mSizeEst3 = params.sampleSizeDouble;
	std::vector<long long int> samp3 = params.sampleSizeDouble;
	int itdxm = 1;
	for(auto mSizeEst3 : samp3)
	{
		//std::cout << "samp3: " << mSizeEst3 << '\n';
		tStart = std::chrono::steady_clock::now();
		estimate = 0;
		gen.seed(params.seed+(100*itdxm));


		std::uniform_int_distribution<> distrib(0, n-1); // [a,b] uniform int in [0,n-1];
		std::vector<long long int> posSelected;
		for(int i=0; i < mSizeEst3; i++)
		{
			posSelected.push_back(distrib(gen));
		}

		//std::vector<long long int> posSelected = sublinearBinomialSample(n, params.sampleSizeFullWithSampl, std::ref(gen));
		//long long int selectedSize = posSelected.size();
		for(auto& pos : posSelected)
		{
			auto vals = GetEstimateTerms(std::cref(weightedSample), std::cref(pointset), pointset[pos], std::cref(clusterAssignments), distanceType);
			double aHat = vals.first;
			double bHat = vals.second;
			double sHat = (bHat - aHat)/(std::max(bHat, aHat));
			estimate += sHat;
		}
		//estimate /= selectedSize;
		estimate /= mSizeEst3;
		tFinish = std::chrono::steady_clock::now();

		totRT = tFinish-tStart;
		res.multipleM.push_back(estimate);
		//res.doubleApproxSilh= estimate;
		res.totRTEst3.push_back(res.stepART + totRT.count());
		itdxm++;
	}
	//exit(0);

	return;
}


double Algorithms::SimplifiedSilhouette(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res)
{
    //Cluster assignments is a vector of size k where each entry contains a vector contaning the indices of the points in the i-th cluter from pointset
    auto tStart = std::chrono::steady_clock::now();
    long long int n = pointset.size();
    auto distanceType = params.distType;
    std::vector<std::vector<long long int>>& clusterAssignments = params.clusterAssignments; // Check if copy
    std::vector<Point<double>>& centers = params.centers;
    int k = clusterAssignments.size();

    double simpl = 0;
    for(int i=0; i < n; i++)
    {
        auto mycluster = pointset[i].getClust();
        double aTerm = 0;
		double bTerm = std::numeric_limits<double>::max();
		double simplSilh = 1;
        if(clusterAssignments[mycluster].size() > 1) // Current cluster has size at most t
		{
			for(int j=0; j< k; j++)
			{
				if(j==mycluster)
				{
					aTerm = ComputeDistancePoints(centers[j], pointset[i], distanceType);
				}
				else
				{
					bTerm = std::min(bTerm, ComputeDistancePoints(centers[j], pointset[i], distanceType));
				}
			}
        	simplSilh = (bTerm-aTerm)/(std::max(aTerm,bTerm));
		}
        simpl += simplSilh; 
    }
    simpl /= n;
    auto tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT1 = tFinish-tStart;
	res.totRTEst1 = totRT1.count();

	res.approxSilh = simpl;

    return simpl;
}

double Algorithms::FrahlingSolhler(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res)
{
    //Cluster assignments is a vector of size k where each entry contains a vector contaning the indices of the points in the i-th cluter from pointset
    auto tStart = std::chrono::steady_clock::now();
    long long int n = pointset.size();
    auto distanceType = params.distType;
    std::vector<std::vector<long long int>>& clusterAssignments = params.clusterAssignments; // Check if copy
    std::vector<Point<double>>& centers = params.centers;
    int k = clusterAssignments.size();

    double simpl = 0;
    for(int i=0; i < n; i++)
    {
        auto mycluster = pointset[i].getClust();
        double aTerm = 0;
		double bTerm = std::numeric_limits<double>::max();
		double minBIndex = -1;
		double simplSilh = 1;
        if(clusterAssignments[mycluster].size() > 1) // Current cluster has size at most t
		{
			for(int j=0; j< k; j++)
			{
				if(j==mycluster)
				{
					aTerm = ComputeDistancePoints(centers[j], pointset[i], distanceType);
				}
				else
				{
					//bTerm = std::min(bTerm, ComputeDistancePoints(centers[j], pointset[i], distanceType));
					if(bTerm > ComputeDistancePoints(centers[j], pointset[i], distanceType))
					{
						bTerm = std::min(bTerm, ComputeDistancePoints(centers[j], pointset[i], distanceType));
						minBIndex = j;
					}
				}
			}
			double avgBTerm=0;
			double avgATerm=0;

			for(int j=0; j < clusterAssignments[minBIndex].size(); j++)
			{
				avgBTerm += ComputeDistancePoints(pointset[clusterAssignments[minBIndex][j]], pointset[i], distanceType);
			}
			avgBTerm /= clusterAssignments[minBIndex].size();

			for(int j=0; j < clusterAssignments[mycluster].size(); j++)
			{
				avgATerm += ComputeDistancePoints(pointset[clusterAssignments[mycluster][j]], pointset[i], distanceType);
			}
			avgATerm /= clusterAssignments[mycluster].size();
			simplSilh = (avgBTerm-avgATerm)/(std::max(avgATerm,avgBTerm));
		}
        simpl += simplSilh; 
    }
    simpl /= n;

    auto tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT1 = tFinish-tStart;
	res.totRTEst1 = totRT1.count();

	res.approxSilh = simpl;

    return simpl;
}

double Algorithms::FastSquaredEuclidSilhouette(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res)
{
    //Cluster assignments is a vector of size k where each entry contains a vector contaning the indices of the points in the i-th cluter from pointset
    auto tStart = std::chrono::steady_clock::now();
    long long int n = pointset.size();
    auto distanceType = params.distType;
    std::vector<std::vector<long long int>>& clusterAssignments = params.clusterAssignments; // Check if copy
    std::vector<Point<double>>& centers = params.centers;
    int k = clusterAssignments.size();
	int dimensions = pointset[0].getDims();
	std::vector<double> squaredEntries(n, 0);

	for(int i=0; i < n; i++)
	{
		for(int j=0; j< dimensions; j++)
		{
			squaredEntries[i] += std::pow(pointset[i].getCoord(j),2);
		}
	}
	std::vector<double> clusterSquaredEntries(k, 0);
	
	for(int i=0; i < k; i++)
	{
		for(int j=0; j < clusterAssignments[i].size(); j++)
		{
			clusterSquaredEntries[i] += squaredEntries[clusterAssignments[i][j]];
		}
	}

	std::vector<Point<double>> entriesSum(k, Point<double>(dimensions));
	for(int i=0; i < k; i++)
	{
		auto& currentSum = entriesSum[i];
		for(int j=0; j < clusterAssignments[i].size(); j++)
		{
			for(int h=0; h < dimensions; h++)
			{
				double updatedVal = pointset[clusterAssignments[i][j]].getCoord(h);
				currentSum.updateCoord(h, updatedVal);
			}
		}
	}

	double silhval = 0;
	for(int i=0; i < n; i++)
	{
		int mycluster = pointset[i].getClust();
        double aTerm = 0;
		double bTerm = std::numeric_limits<double>::max();
		for(int j=0; j< k; j++)
		{
			double avgDistance = clusterSquaredEntries[j];
			double lastTerm = 0;
			for(int h=0; h < dimensions; h++)
			{
				lastTerm += entriesSum[j].getCoord(h) * pointset[i].getCoord(h);
			}
			avgDistance -= 2*lastTerm;
			avgDistance /= clusterAssignments[j].size();
			avgDistance += squaredEntries[i];
			if(j==mycluster)
			{
				if(clusterAssignments[j].size() > 1)
				{
					//aTerm = avgDistance;
					aTerm = (clusterAssignments[j].size()*squaredEntries[i] + clusterSquaredEntries[j] - (2*lastTerm))/(clusterAssignments[j].size()-1);
				}
				else
					aTerm = 0;
			}
			else
			{
				bTerm = std::min(bTerm, avgDistance);
			}
		}
		double silhPoint = (bTerm-aTerm)/std::max(bTerm, aTerm);
		silhval += silhPoint;
	}
	silhval /= n;


    auto tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT1 = tFinish-tStart;
	res.totRTEst1 = totRT1.count();

	res.approxSilh = silhval;

	return silhval;
}


void Algorithms::ComputeEstimatorSubsample(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res)
{
    auto tStart = std::chrono::steady_clock::now();
    std::mt19937_64 gen;
    gen.seed(params.seed);
    long long int n = pointset.size();
    std::vector<std::vector<long long int>>& clusterAssignments = params.clusterAssignments; 
    auto distanceType = params.distType;
    int k = clusterAssignments.size();

	std::vector<long long int> clustSizes(k,0);
	for(auto& p : pointset) // O(n)
	{
		clustSizes[p.getClust()]++;
	}

	int mSizeEst1 = k*params.t;
	double estimate = 0;
	std::uniform_int_distribution<> distrib(0, n-1); // [a,b] uniform int in [0,n-1];
	std::vector<long long int> posSelected;
	for(int i=0; i < mSizeEst1; i++)
	{
		posSelected.push_back(distrib(gen));
	}
		
	for(auto i : posSelected)
	{
		auto x = pointset[i];
		auto myClust = x.getClust();
		std::vector<double> averageDistances(k,0);

		for(int j=0; j < pointset.size(); j++)
		{
			if(i!=j)
			{
				averageDistances[pointset[j].getClust()] += ComputeDistancePoints(x, pointset[j], distanceType);
			}
		}
		double a = 0;
		double b = std::numeric_limits<double>::max();
		for(int j=0; j < k; j++)
		{
			if(j!= myClust)
			{
				if(clustSizes[j] > 0)
				{
					averageDistances[j] /= clustSizes[j];
					b = std::min(b, averageDistances[j]);
				}
			}
		}

		if(clustSizes[myClust] > 1)
			a = averageDistances[myClust]/(clustSizes[myClust]-1);
		else
			a = 0;

		double silh = (b-a)/(std::max(b, a));
		estimate += silh;
	}
	estimate = estimate/mSizeEst1;

    auto tFinish = std::chrono::steady_clock::now();
    std::chrono::duration<double> totRT1 = tFinish-tStart;
	res.totRTEst1 = totRT1.count();

	res.approxSilh = estimate;
	return;
}

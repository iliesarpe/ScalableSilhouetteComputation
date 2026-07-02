#ifndef ALGS_H
#define ALGS_H

#include "structures.h"
#include <random>
#include <unordered_set>
#include <functional>

using Structures::Point;
using Structures::AlgorithmParameters;
using Structures::AlgorithmResult;

#define EUCLID 0
#define SQUARED 1
#define MANHATTAN 2
#define COSINE 3
#define CANBERRA 4
#define ALLPOINTS 1
#define FULLNOSAMPL 2
#define FULLWITHSAMPL 3

namespace Algorithms {
   double KMeans(std::vector<Point<double>>& pointset, std::vector<Point<double>>& centroids, int k, int iterations, bool kmeansPP=true, long long int seed=100);
   std::vector<Point<double>> InitializeCentersKMeans(std::vector<Point<double>>& pointset, int k, bool kmeansPP, long long int seed);
   //Type =0, L2squared norm
   inline double ComputeDistancePoints(const Point<double>& p, const Point<double>& p2, int type=EUCLID)
   {
       double distance = 0;
       if(type==EUCLID)
           distance = Structures::EuclidDist(p, p2);
       else if(type==SQUARED)
           distance = Structures::EuclidSquaredDist(p, p2);
       else if(type==MANHATTAN)
           distance = Structures::ManhattanDist(p, p2);
       else if(type==COSINE)
           distance = Structures::CosineDist(p, p2);
       else if(type==CANBERRA)
           distance = Structures::CanberraDist(p, p2);
       else
           std::cerr << "Unknown distance type!!" << std::endl;
       return distance;
   }

   inline std::pair<double, double> GetEstimateTerms(const std::vector<std::vector<std::pair<long long int, double>>>& weightedSample, const std::vector<Point<double>>& pointset, Point<double> currentPoint,const std::vector<std::vector<long long int>>& clusterAssignments, int distanceType)
    {
        auto myclust = currentPoint.getClust();
		int k = weightedSample.size();
        double bHat = std::numeric_limits<double>::max();
        double aHat = 0;
        for(int j=0; j < k; j++)
        {
            auto proxyPoints = weightedSample[j];
            double weightClust = 0;
            for(auto& pair : proxyPoints)
            {
                weightClust += (ComputeDistancePoints(pointset[pair.first], currentPoint, distanceType) * (1./pair.second));
            }
            if(myclust != j)
            {
                weightClust /= clusterAssignments[j].size();
                bHat = std::min(bHat, weightClust);
            }
            else
            {
				if(clusterAssignments[myclust].size() > 1)
                	aHat = weightClust/((1.*clusterAssignments[myclust].size()) -1.);
				else
					aHat = 0;
            }
        }
        return std::make_pair(aHat, bHat);
    }

	std::vector<long long int> sublinearBinomialSample(long long int trials, double probability, std::mt19937_64& generator);

	// Exact
	std::vector<double> ComputeExactSilhouettePoints(std::vector<Point<double>>& pointset, int k, int distance=SQUARED);

	//Approx PPS and uni
	void ApproximateSilhPPS(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res);
	void ApproximateSilhUniform(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res);
	//Computing global Estimator1
	void ComputeEstimatorSubsample(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res);

	// Baselines, Simplified, FrahSol, and FastSquaredEuclid
	double SimplifiedSilhouette(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res);
	double FrahlingSolhler(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res);
	double FastSquaredEuclidSilhouette(std::vector<Point<double>>& pointset, AlgorithmParameters& params, AlgorithmResult& res);

	std::vector<double> ComputeErrorBuckets(const std::vector<double>& exSilh, const std::vector<double>& appxSilh, const std::vector<int>& pointBuckets, int totBucks);
	std::vector<std::vector<double>> ComputeErrorClusters(const std::vector<double>& exSilh, const std::vector<Point<double>>& pointset, const std::vector<double>& appxSilh,int k);


}


#endif //ALGS_H

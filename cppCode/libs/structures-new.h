#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <vector>
#include <iostream>
#include <functional>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include <highfive/H5Easy.hpp>
#include <chrono>

#define DIVIDE 1
#define SUM 0
#define KMEANSM 0
#define KMEDOIDSM 1

using json = nlohmann::json;

namespace Structures {
    template <typename T> class Point{
    private: 
        std::vector<T> coords;
        int clust = -1; // keeps the index of the cluster the point is assigned to, 0,...,k-1
                   // See if we want also to keep closest centroid ecc
    public:
        Point(int dimension) {
            coords.resize(dimension, 0);
        }
        Point(std::vector<T> c) {
            //coords.resize(c.size());
			coords.resize(0);
            coords.insert(coords.begin(), c.begin(), c.end());
        }
        T getCoord(int pos) const {
            return coords[pos];
        }
        const std::vector<T> getAllCoords() const {
			return std::cref(coords);	
        }
        void setCoord(int pos, T val){
            coords[pos] = val;
            return;
        }
        void updateCoord(int pos, T val, int operation=SUM){
            if(operation == SUM)
            {
                coords[pos] += val;
            }
            else if(operation == DIVIDE)
                coords[pos] /= val;
            return;
        }
        void setClust(int myc){
            clust = myc;
            return;
        }
        int getClust() const {
            return clust;
        }
        int getDims() const {
            return coords.size();
        }

    };
    
    struct AlgorithmParameters{
        std::vector<Point<double>> centers;
        std::vector<std::vector<long long int>> clusterAssignments;
        int t; 
		int k;
		long long int seed;
        double varepsilon;
        double delta;
        int mode; //0: local estimator
        int distType; //0: local estimator
        int threads=1;
        double sampleSizeFullWithSampl; //0: local estimator
        //long long int sampleSizeDouble; 
        std::vector<long long int> sampleSizeDouble; 
    };

	struct AlgorithmResult{
		std::vector<double> silhApproxValues; // approx value for each point
		double approxSilh;
		double doubleApproxSilh;
		double totRTEst1;
		std::vector<double> totRTEst3;
		std::vector<double> multipleM;
		double stepART;
    };
    
    // We can consider other distances, e.g., the ones where the silhouette cannot be computed efficiently with guarantees
    inline double EuclidSquaredDist(const Point<double>& p1, const Point<double>& p2)
    {
        auto dims = p1.getDims();
        double dist = 0;
        for(int i = 0; i < dims; i++)
        {
            double diff = p1.getCoord(i) - (1.0* p2.getCoord(i));
            dist += std::pow(diff, 2);
        }
        return dist;
    }

    inline double EuclidDist(const Point<double>& p1, const Point<double>& p2)
    {
        return std::sqrt(EuclidSquaredDist(p1,p2));
    }

	inline double ManhattanDist(const Point<double>& p1, const Point<double>& p2)
    {
        auto dims = p1.getDims();
        double dist = 0;
        for(int i = 0; i < dims; i++)
        {
            double diff = std::abs(p1.getCoord(i) - (1.0* p2.getCoord(i)));
            dist += diff;
        }
        return dist;
    }

	inline double CosineDist(const Point<double>& p1, const Point<double>& p2)
    {
        auto dims = p1.getDims();
		double norm1=0;
		double norm2=0;
        double dist = 0;
        for(int i = 0; i < dims; i++)
        {
            double term = p1.getCoord(i) * (1.* p2.getCoord(i));
			norm1 += std::pow(p1.getCoord(i), 2);
			norm2 += std::pow(p2.getCoord(i), 2);
            dist += term;
        }
		norm1 = std::pow(norm1, 0.5);
		norm2 = std::pow(norm2, 0.5);
		dist /= (norm1*norm2);
		dist = 1.-dist;
        return dist;
    }

	inline double CanberraDist(const Point<double>& p1, const Point<double>& p2)
    {
        auto dims = p1.getDims();
        double dist = 0;
        for(int i = 0; i < dims; i++)
        {
			if((std::abs(p1.getCoord(i)) + std::abs(p2.getCoord(i))) == 0) ;
			else
			{
            	double diff = std::abs(p1.getCoord(i) - (1.0* p2.getCoord(i)))/( std::abs(p1.getCoord(i)) + std::abs(p2.getCoord(i)) );
            	dist += diff;
			}
        }
		if (std::isinf(dist))
		{
			std::cout << "Is inf:" << std::endl;
			dist = 0;
		}
		if(std::isnan(dist)) // 0/0
		{
			/*
			std::cout << "Is nan:" << std::endl;
			for(int i = 0; i < dims; i++)
			{
				std::cout << "P1-dim i-" << i << " has: " << p1.getCoord(i) << " p2: " <<  p2.getCoord(i) << std::endl;
			}
			exit(0);
			*/
			dist = 0;
		}
        return dist;
    }

    std::vector<Structures::Point<double>> BuildDatasetRaw(std::string fname);
    std::vector<Structures::Point<double>> BuildDatasetRaw(H5Easy::File& datafile, AlgorithmParameters& params);
	void LoadClustering(H5Easy::File& datafile, AlgorithmParameters& params, std::string path, std::vector<Structures::Point<double>>& pointset, std::string alg);
	void dumpHDF5(std::vector<Point<double>>& pointset, H5Easy::File& outfile, int type=0);

	// ---- General-purpose (non-experiment) I/O for standalone/library usage ----
	// Reads a flat cluster-assignment vector: one integer cluster id (0..k-1) per line/row.
	std::vector<int> LoadLabelsCSV(std::string fname);
	std::vector<int> LoadLabelsHDF5(H5Easy::File& datafile, std::string path);
	// Turns a flat label vector into the per-cluster index lists AlgorithmParameters expects.
	// Validates that every label is in [0,k) and exits with a clear error otherwise.
	std::vector<std::vector<long long int>> BuildClusterAssignments(const std::vector<int>& labels, int k);
}
//
// Structures implementation

#endif //STRUCTURES_H

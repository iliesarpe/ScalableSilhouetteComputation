#include "structures.h"

std::vector<Structures::Point<double>> Structures::BuildDatasetRaw(std::string fname)
{
	std::cerr << "Reading graph from text file " << fname << " " << std::endl;
    std::ifstream myfile(fname);
	std::string line = "";
	std::vector<Point<double>> dataset;
    while(std::getline(myfile, line))
	{
        std::istringstream iss{line};
		std::string coord ="";
		std::vector<double> coords;
		int dims = 0;
		while(std::getline(iss, coord, ','))
		{
			coords.push_back(std::stof(coord));
			dims++;
		}
		coords.resize(dims);
		Point currP = Point(coords);
		dataset.push_back(currP);
	}
	return dataset;
}

#ifdef USE_HDF5
std::vector<Structures::Point<double>> Structures::BuildDatasetRaw(H5Easy::File& datafile, AlgorithmParameters& params)
{
	std::vector<std::vector<double>> pointsRaw = H5Easy::load<std::vector<std::vector<double>>>(datafile, "dataset");
	std::vector<Structures::Point<double>> pointset;
	for(auto& rawPoint : pointsRaw)
	{
		Point currP = Point(rawPoint);
		pointset.push_back(currP);
	}
	/*
	std::vector<int> assignment = H5Easy::load<std::vector<int>>(datafile, "clustering/kmeans/k-9/assignment");
	for(auto& val : assignment)
	{
		std::cout << "Value is : " << val << '\n';
	}
	pointsRaw.clear();
	pointsRaw = H5Easy::load<std::vector<std::vector<double>>>(datafile, "clustering/kmeans/k-9/centers");
	for(auto& rawPoint : pointsRaw)
	{
		std::cout << "centroid: ";
		for(auto& e : rawPoint)
		{
			std::cout << e << ",";
		}
		std::cout << std::endl;
	}
	*/

	return pointset;
}

void Structures::LoadClustering(H5Easy::File& datafile, AlgorithmParameters& params, std::string path, std::vector<Structures::Point<double>>& pointset, std::string alg)
{
	std::string path1 = path + "/assignment";
	std::vector<int> assignment = H5Easy::load<std::vector<int>>(datafile, path1);
	long long int currentPoint = 0;
	params.clusterAssignments.clear();
	params.centers.clear();
	params.clusterAssignments.resize(params.k);
	for(auto& clust : assignment)
	{
		params.clusterAssignments[clust].push_back(currentPoint);
		currentPoint++;
	}
	if(alg == "kmeans")
	{
		std::string path2 = path + "/centers";
		std::vector<std::vector<double>> pointsRaw = H5Easy::load<std::vector<std::vector<double>>>(datafile, path2);
		for(auto rawPoint : pointsRaw)
		{
			Point currP = Point(rawPoint);
			params.centers.push_back(currP);
		}
	}
	else if (alg == "kmedoids")
	{
		std::string path2 = path + "/centers";
		std::vector<int> pointsIdx = H5Easy::load<std::vector<int>>(datafile, path2);
		for(auto idx : pointsIdx)
		{
			Point currP = pointset[idx];
			params.centers.push_back(currP);
		}
	}

	return;
}
#endif // USE_HDF5

std::vector<int> Structures::LoadLabelsCSV(std::string fname)
{
	std::ifstream myfile(fname);
	if(!myfile.is_open())
	{
		std::cerr << "ERROR: cannot open assignment file " << fname << std::endl;
		exit(1);
	}
	std::vector<int> labels;
	std::string line = "";
	while(std::getline(myfile, line))
	{
		if(line.empty())
			continue;
		labels.push_back(std::stoi(line));
	}
	return labels;
}

#ifdef USE_HDF5
std::vector<int> Structures::LoadLabelsHDF5(H5Easy::File& datafile, std::string path)
{
	return H5Easy::load<std::vector<int>>(datafile, path);
}
#endif // USE_HDF5

std::vector<std::vector<long long int>> Structures::BuildClusterAssignments(const std::vector<int>& labels, int k)
{
	std::vector<std::vector<long long int>> clusterAssignments(k);
	for(long long int i = 0; i < static_cast<long long int>(labels.size()); i++)
	{
		int c = labels[i];
		if(c < 0 || c >= k)
		{
			std::cerr << "ERROR: label " << c << " at point " << i
			           << " is out of range [0," << (k-1) << "]. "
			           << "Cluster ids must be 0-indexed and contiguous; pass \"k\" explicitly "
			           << "if it should not be inferred from max(labels)+1." << std::endl;
			exit(1);
		}
		clusterAssignments[c].push_back(i);
	}
	return clusterAssignments;
}



#ifdef USE_HDF5
//void dumpHDF5(std::vector<Structures::Point<double>>& pointset, std::string outfilename)
void Structures::dumpHDF5(std::vector<Structures::Point<double>>& pointset, H5Easy::File& outfile, int type)
{

	H5Easy::DumpOptions options{H5Easy::Compression()};
	if(type == 1)
	{
		std::cerr << "Dumping points" << std::endl;
		std::vector<std::vector<double>> points;
		for(auto& point : pointset)
		{
			std::vector<double> mycoords = point.getAllCoords();
			points.push_back(mycoords);
		}
		H5Easy::dump(outfile, "/points/coords", points, options);
	}
	return;
	/*
	std::cerr << "Dumping points" << std::endl;
	std::vector<std::vector<double>> points;
	std::vector<
	//sort_by_tim();
	H5Easy::dump(file, "/edges/timestamp-interactions", edges, options);
	std::cerr << "Dumping id map" << std::endl;
	H5Easy::dump(file, "/meta/id-map", new_old_remapping, options);
	std::cerr << "Dumping static out degrees" << std::endl;

	H5Easy::dumpAttribute(file, "/meta/id-map", "Nodes", N);
	std::cerr << "Dumping the number of nodes" << std::endl;

	H5Easy::dump(file, "/statistics/static/degree/out", static_out_degrees(), options);
	std::cerr << "Dumping static in degrees" << std::endl;
	H5Easy::dump(file, "/statistics/static/degree/in", static_in_degrees(), options);
	std::cerr << "Dumping temporal out degrees" << std::endl;
	H5Easy::dump(file, "/statistics/temporal/degree/out", temporal_out_degrees(), options);
	std::cerr << "Dumping temporal in degrees" << std::endl;
	H5Easy::dump(file, "/statistics/temporal/degree/in", temporal_in_degrees(), options);
	std::cerr << "Done" << std::endl;
	*/

}
#endif // USE_HDF5

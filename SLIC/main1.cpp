#include<pcl/point_cloud.h>
#include<pcl/filters/uniform_sampling.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<vector>
#include<iostream>
#include<cmath>
#include<random>
#include<string>
#include<boost/program_options.hpp>
#include<fstream>

#include "evaluation.hpp"
#include "SLIC.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace pcl;


typedef pcl::PointXYZ	PointT;
typedef pcl::PointXYZRGB	PointTC;
typedef pcl::PointXYZRGBL		PoinTL;



int main()
{
	pcl::PointCloud<PoinTL>::Ptr cloud(new pcl::PointCloud<PoinTL>);
	pcl::PointCloud<pcl::PointXYZL>::Ptr result(new pcl::PointCloud<PointXYZL>);
	io::loadPCDFile("D:\\data\\s3dis\\S3dis_a1_office17.pcd", *cloud);
	ofstream ofs;
	ofs.open("result.csv", ios::out);
	SLIC<PoinTL> slic(cloud);
	slic.setL2_min(10.0);

	float userr, asa;

	for (int i = 1; i < 101; i++)
	{
		for (int j = 1; j < 11; j++)
		{
			cout << 0.1 * float(i) << " " << 0.01 * float(j) << endl;
			slic.setS(0.1 * float(i));
			slic.setM(0.1 * float(j));
			slic.SLIC_superpointcloudclusting();
			slic.getLabledCloud(result);

			userr = Cal_undersegmentation_error(result, cloud);
			asa = Cal_Achievable_seg_acc(result, cloud);
			ofs << 0.1 * float(i) << "," << 0.01 * float(j) << "," << slic.getSuperpixelCount() << "," << userr << "," << asa << std::endl;
		}
	}
	ofs.close();
	return 0;
}
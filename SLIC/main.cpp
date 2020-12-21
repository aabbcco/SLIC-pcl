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
#include<functional>

#include "evaluation.hpp"
#include "SLIC.hpp"

namespace po = boost::program_options;
using namespace std;


typedef pcl::PointXYZ	PointT;
typedef pcl::PointXYZRGB	PointTC;
typedef pcl::PointXYZRGBL		PoinTL;



int main(int argc, char* argv[])
{
	float s = 10.0f;
	float m = 1.0f;
	float L2_min = 10.0f;
	std::string dataroot;

	//argparse`
	po::options_description opts("All opts");
	po::variables_map vm;
	opts.add_options()
		("search,s", po::value<float>(&s)->default_value(1.0f), "Filtering and search radius s")
		("importance,m", po::value<float>(&m)->default_value(1.0f), "Spital importance m")
		("l2,l",po::value<float>(&L2_min)->default_value(10.0f),"minium L2 loss")
		("dataroot,d",po::value<std::string>(&dataroot)->default_value("F:\\aabbcco\\vscode_python\\single_python\\shapenet.pcd"),"data to be segmented")
		("help,h", "SLIC like Superpixel using PCL Library");
	try
	{
		po::store(po::parse_config_file("config.cfg", opts), vm);
		po::notify(vm);
	}
	catch (...)
	{
		std::cout << "wrong input" << endl;
		return -1;
	}

	if (vm.count("help"))
	{
		std::cout << opts << std::endl;
		return 0;
	}


	std::cout << "SLIC step for "<<dataroot<<" using s=" << s << " m=" << m << " L2 min=" << L2_min << std::endl;

	pcl::PointCloud<PoinTL>::Ptr cloud(new pcl::PointCloud<PoinTL>);
	pcl::io::loadPCDFile(dataroot.c_str(), *cloud);

	pcl::PointCloud<PoinTL>::Ptr clusting_center(new pcl::PointCloud<PoinTL>);
	pcl::PointCloud<pcl::PointXYZL>::Ptr labledcloud(new pcl::PointCloud<pcl::PointXYZL>);

	//auto sam = std::bind(RandomSampling<PoinTL>, std::placeholders::_2, 50);
	SLIC<PoinTL> clusting(cloud,m,s,L2_min);
	//clusting.setSamplingFunc(sam);
	clock_t clk1, clk2 = clock();
	clusting.SLIC_superpointcloudclusting();
	clk1 = clock();
	clusting.getLabledCloud(labledcloud);
	clusting.getSeed(clusting_center);
	
	float error = Cal_undersegmentation_error(labledcloud, cloud,true);
	float asa = Cal_Achievable_seg_acc(labledcloud, cloud);

	std::cout << "Under-segmentation error is: " << error <<" ASA: "<<asa<<" using time :"<<clk1-clk2<<"ms"<< std::endl;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr  orig(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*cloud, *orig);
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("SLIC"));
	int v1(0), v2(1),v3(2);
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->addPointCloud(orig, "origional cloud", v1);
	viewer->createViewPort(0.5, 0,1, 1, v2);
	viewer->addPointCloud(labledcloud,"labled cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "origional cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "labled cloud", v2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr center(new pcl::PointCloud<pcl::PointXYZ>);
	
	pcl::copyPointCloud(*clusting_center, *center);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(center, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(center, "clusting center1", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "clusting center1");
	
	viewer->spin();

	return 0;
}
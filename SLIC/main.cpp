#include<pcl/point_cloud.h>
#include<pcl/filters/uniform_sampling.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<vector>
#include<iostream>
#include<cmath>
#include<random>
#include<boost/program_options.hpp>

#include "evaluation.hpp"
#include "SLIC.hpp"

namespace po = boost::program_options;


typedef pcl::PointXYZ	PointT;
typedef pcl::PointXYZRGB	PointTC;
typedef pcl::PointXYZL		PoinTL;



int main(int argc, char* argv[])
{
	float s = 10.0f;
	float m = 1.0f;
	float L2_min = 10.0f;

	//argparse
	po::options_description opts("All opts");
	po::variables_map vm;
	opts.add_options()
		("s", po::value<float>()->default_value(10.0f), "Filtering and search radius s")
		("m", po::value<float>()->default_value(1.0f), "Spital importance m")
		("l2",po::value<float>()->default_value(10.0f),"minium L2 loss")
		("help", "SLIC like Superpixel using PCL Library");
	try
	{
		po::store(po::parse_command_line(argc, argv, opts), vm);
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

	s = vm["s"].as<float>();
	m = vm["m"].as<float>();
	L2_min = vm["l2"].as<float>();

	std::cout << "SLIC step using s=" << s << " m=" << m << " L2 min=" << L2_min << std::endl;

	pcl::PointCloud<PoinTL>::Ptr cloud(new pcl::PointCloud<PoinTL>);
	pcl::io::loadPCDFile("C:\\Users\\37952\\Desktop\\000_orig_pcd\\benthi_control_A_D30_centre_filter.pcd", *cloud);

	pcl::PointCloud<PoinTL>::Ptr clusting_center(new pcl::PointCloud<PoinTL>);
	pcl::PointCloud<pcl::PointXYZL>::Ptr labledcloud(new pcl::PointCloud<pcl::PointXYZL>);

	SLIC<PoinTL> clusting(cloud,m,s,L2_min);
	clusting.SLIC_superpointcloudclusting();
	clusting.getLabledCloud(labledcloud);
	clusting.getSeed(clusting_center);
	
	float error = Cal_undersegmentation_error(labledcloud, cloud, 26, clusting.getSuperpixelCount(),true);
	float asa = Cal_Achievable_seg_acc(labledcloud, cloud, 26, clusting.getSuperpixelCount());

	std::cout << "Under-segmentation error is: " << error <<" ASA: "<<asa<< std::endl;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("SLIC"));
	int v1(0), v2(1),v3(2);
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->addPointCloud(cloud, "origional cloud", v1);
	viewer->createViewPort(0.5, 0,1, 1, v2);
	viewer->addPointCloud(labledcloud,"labled cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "origional cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "labled cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "labled cloud", v2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr center(new pcl::PointCloud<pcl::PointXYZ>);
	
	pcl::copyPointCloud(*clusting_center, *center);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(center, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(center, "clusting center1", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "clusting center1");
	
	viewer->spin();

	return 0;
}
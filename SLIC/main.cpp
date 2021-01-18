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
#include<memory>
#include<pcl/features/normal_3d_omp.h>

#include "evaluation.hpp"
#include "SLIC.hpp"

using namespace pcl;
namespace po = boost::program_options;
using namespace std;


typedef PointXYZ			PointT;
typedef PointNormal			PointTN;
typedef PointXYZRGB			PointTC;
typedef PointXYZL		PoinTL;
typedef PointXYZINormal		PoinTIN;



int main(int argc, char* argv[])
{
	float s = 10.0f;
	float m = 1.0f,mn,mc;
	float L2_min = 10.0f;
	std::string dataroot;

	//argparse`
	po::options_description opts("All opts");
	po::variables_map vm;
	opts.add_options()
		("search,s", po::value<float>(&s)->default_value(1.0f), "Filtering and search radius s")
		("importanceN", po::value<float>(&mn)->default_value(1.0f), "Spital importance m")
		("importanceC", po::value<float>(&mc)->default_value(1.0f), "Spital importance m")
		("importance", po::value<float>(&m)->default_value(1.0f), "Spital importance m")
		("l2,l",po::value<float>(&L2_min)->default_value(10.0f),"minium L2 loss")
		("dataroot,d",po::value<std::string>(&dataroot)->default_value("F:\\aabbcco\\vscode_python\\single_python\\shapetest\\0.pcd"),"data to be segmented")
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

	PointCloud<PointT>::Ptr cloud(new PointCloud<PointT>);
	io::loadPCDFile(dataroot.c_str(), *cloud);

	PointCloud<PointTN>::Ptr clusting_center(new PointCloud<PointTN>);
	PointCloud<PointXYZL>::Ptr labledcloud(new PointCloud<PointXYZL>);
	PointCloud<PointTN>::Ptr NCloud(new PointCloud<PointTN>);
	PointCloud<Normal>::Ptr Ncloud(new PointCloud<Normal>);
	
	copyPointCloud(*cloud, *NCloud);
	NormalEstimationOMP<PointT, Normal> nest;
	nest.setInputCloud(cloud);
	nest.setKSearch(30);
	search::KdTree<PointT>::Ptr tree(new search::KdTree<PointT>);
	nest.setSearchMethod(tree);
	nest.compute(*Ncloud);
	for (auto i = 0; i < Ncloud->width; i++)
	{
		memcpy(&(NCloud->points[i].data_n), &(Ncloud->points[i].data_n), 4 * sizeof(float));
		NCloud->points[i].curvature = Ncloud->points[i].curvature;
	}
/*
	visualization::PCLVisualizer::Ptr viewer1(new visualization::PCLVisualizer);

	viewer1->addPointCloud(cloud, "ncloud");
	viewer1->addPointCloudNormals<PointT, Normal>(cloud, Ncloud,1,0.05f,"nncloud");
	viewer1->spin();
*/

	//auto sam = std::bind(RandomSampling<PoinTL>, std::placeholders::_2, 50);
	//Normal
	SLIC<PointTN> clusting(NCloud,mn,s,L2_min);
	clock_t clk1, clk2 = clock();
	clusting.SLIC_superpointcloudclusting();
	clk1 = clock();
	clusting.getLabledCloud(labledcloud);
	clusting.getSeed(clusting_center);
	//curvature
	PointCloud<PoinTIN>::Ptr TINcloud(new PointCloud<PoinTIN>);
	copyPointCloud(*NCloud, *TINcloud);
	unique_ptr<typename SLIC<PoinTIN>> clusting1(new SLIC<PoinTIN>(TINcloud,mc,s,L2_min));
	clusting1->SLIC_superpointcloudclusting();
	PointCloud<PoinTL>::Ptr TINlabel(new PointCloud<PoinTL>);
	clusting1->getLabledCloud(TINlabel);
	//the most common XYZ cloud
	unique_ptr<typename SLIC<PointT>> clusting2(new SLIC<PointT>(cloud, m, s, L2_min));
	PointCloud<PoinTL>::Ptr labelT(new PointCloud<PoinTL>);
	clusting2->SLIC_superpointcloudclusting();
	clusting2->getLabledCloud(labelT);


	int v[] = { 1,2,3,4 };
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer);
	for(auto i=0;i<4;i++)
	{
		viewer->createViewPort((i/2) *0.5,(i%2)*0.5,(i/2+1)*0.5, (i % 2+1) * 0.5,v[i]);
	}
	viewer->addPointCloud(cloud, "origion", v[0]);
	viewer->addPointCloud(TINlabel, "TINlabel", v[1]);
	viewer->addPointCloud(labledcloud, "labledcloud", v[2]);
	viewer->addPointCloud(labelT, "labelT", v[3]);
	viewer->addText("origion", 0, 0, "origion",v[0]);
	viewer->addText("curvature", 0, 0, "TINlabel", v[1]);
	viewer->addText("normal", 0, 0, "labledcloud", v[2]);
	viewer->addText("origion", 0, 0, "labelT", v[3]);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 13, "origion", v[0]);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 13, "TINlabel", v[1]);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 13, "labledcloud", v[2]);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 13, "labelT", v[3]);

	viewer->spin();


/*
	float error = Cal_undersegmentation_error(labledcloud, cloud,true);
	float asa = Cal_Achievable_seg_acc(labledcloud, cloud);

	std::cout << "Under-segmentation error is: " << error <<" ASA: "<<asa<<" using time :"<<clk1-clk2<<"ms"<< std::endl;

	PointCloud<PointXYZRGB>::Ptr  orig(new PointCloud<PointXYZRGB>);
	copyPointCloud(*cloud, *orig);
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("SLIC"));
	int v1(0), v2(1),v3(2);
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->addPointCloud(orig, "origional cloud", v1);
	viewer->createViewPort(0.5, 0,1, 1, v2);
	viewer->addPointCloud(labledcloud,"labled cloud", v2);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "origional cloud", v1);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "labled cloud", v2);
	PointCloud<PointXYZ>::Ptr center(new PointCloud<PointXYZ>);
	
	copyPointCloud(*clusting_center, *center);
	visualization::PointCloudColorHandlerCustom<PointXYZ> single_color(center, 255, 255, 255);
	viewer->addPointCloud<PointXYZ>(center, "clusting center1", v1);
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, "clusting center1");
	
	viewer->spin();
*/
	return 0;
}
#include<pcl/point_cloud.h>
#include<pcl/filters/random_sample.h>
#include<pcl/filters/uniform_sampling.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<vector>
#include<iostream>
#include<cmath>
#include<random>

template<typename PointTT>
class SLIC
{
public:
//	constructor/deconstructor
	SLIC() {};
	SLIC(const typename pcl::PointCloud<PointTT>::Ptr &cloud, float m = 1.0f, float s = 1.0f, float L2_min = 1.0f) :m(m), s(s), L2_min(L2_min)
	{
		this->cloud = cloud;
	};
	~SLIC() {};

	//socket in
	void setM(float m)
	{
		this->m = m;
	}
	void setS(float s)
	{
		this->s = s;
	}
	void setL2_min(float L2_min)
	{
		this->L2_min = L2_min;
	}
	void setInputCloud(pcl::PointCloud<PointTT> &cloud)
	{
		this->cloud = cloud;
	}
	//socket out
	float getM() { return m; }
	float getS() { return s; }
	float geL2_min() { return L2_min; }

	void getSeed(const typename pcl::PointCloud<PointTT>::Ptr &seed);
	void getLabledCloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr &result);
	void getLabledCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &result);
	//actions
	void SLIC_superpointcloudclusting();

private:
	float m;
	float s;
	float L2_min;

	typename pcl::PointCloud<PointTT>::ConstPtr cloud;
	typename pcl::PointCloud<PointTT>::ConstPtr seed;

	float Calculate_SLIC_dist(const PointTT &cloud, float sp_dist);

	void filterClustingSeed();

	std::vector<int> label;
};



typedef pcl::PointXYZ	PointT;
typedef pcl::PointXYZRGB	PointTC;
typedef pcl::PointXYZL		PoinTL;



int main(int argc, char* argv[])
{
	float s = 0.1f;
	float m = 0.5f;
	//int samples = 800;
	float L2_min = 1.0f;

	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::io::loadPCDFile("C:\\Users\\37952\\Documents\\toys\\d1_denoise_new.pcd", *cloud);

	pcl::PointCloud<PointT>::Ptr clusting_center(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::PointXYZL>::Ptr labledcloud(new pcl::PointCloud<pcl::PointXYZL>);

	SLIC<pcl::PointXYZ> clusting(cloud,0.5f,0.1f,0.01f);
	clusting.SLIC_superpointcloudclusting();
	clusting.getLabledCloud(labledcloud);
	clusting.getSeed(clusting_center);
	

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("SLIC"));
	int v1(0), v2(1),v3(2);
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->addPointCloud(cloud, "origional cloud", v1);
	viewer->createViewPort(0.5, 0,1, 1, v2);
	viewer->addPointCloud(labledcloud,"labled cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "origional cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "labled cloud", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "labled cloud", v2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr center(new pcl::PointCloud<pcl::PointXYZ>);
	
	pcl::copyPointCloud(*clusting_center, *center);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(center, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(center, "clusting center1", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "clusting center1");
	
	viewer->spin();

	return 0;
}






template<typename PointTT>
void SLIC<PointTT>::SLIC_superpointcloudclusting()
{
	std::cout << "clusting start" << std::endl;
	float x, y, z, L2;
	float count = 0.0f;
	std::vector<float> slic_dist(cloud->width, 1000.0f);
	typename pcl::PointCloud<PointTT>::Ptr clusting_center(new pcl::PointCloud<PointTT>);
	typename pcl::PointCloud<PointTT>::Ptr new_clusting_center(new pcl::PointCloud<PointTT>);
	filterClustingSeed();
	pcl::copyPointCloud(*seed, *new_clusting_center);
	std::vector<int>search_indices;
	std::vector<float>point_square_dist;

	pcl::KdTreeFLANN<PointTT> kd_tree;
	kd_tree.setInputCloud(cloud);

	do
	{
		label = std::vector<int>(cloud->width, 0);
		slic_dist = std::vector<float>(cloud->width, 1000.0f);
		pcl::copyPointCloud(*new_clusting_center, *clusting_center);
		for (int i = clusting_center->width - 1; i >= 0; i--)//each seed point
		{
			search_indices.clear();
			point_square_dist.clear();
			kd_tree.radiusSearch((*clusting_center)[i], 2 * s, search_indices, point_square_dist, 0);
			x = y = z = 0.0f; //clear x,y,z
			count = 0.0f;
			for (int j = search_indices.size() - 1; j >= 0; j--)//each point in 2S radius
			{
				float dist_slic = Calculate_SLIC_dist((*cloud)[search_indices[j]], point_square_dist[j]);

				if (dist_slic < slic_dist[search_indices[j]])
				{
					slic_dist[search_indices[j]] = dist_slic;
					label[search_indices[j]] = i;
					x += (*cloud)[search_indices[j]].x;
					y += (*cloud)[search_indices[j]].y;
					z += (*cloud)[search_indices[j]].z;
					count += 1.0f;
				}
			}
			//compute new seed
			if (count != 0)
			{
				(*new_clusting_center)[i].x = x / count;
				(*new_clusting_center)[i].y = y / count;
				(*new_clusting_center)[i].z = z / count;
			}
		}
		//compute L2 norm
		L2 = 0.0f;
		for (int i = clusting_center->width - 1; i >= 0; i--)
		{
			L2 += ((*new_clusting_center)[i].x - (*clusting_center)[i].x)*((*new_clusting_center)[i].x - (*clusting_center)[i].x)
				+ ((*new_clusting_center)[i].y - (*clusting_center)[i].y)*((*new_clusting_center)[i].y - (*clusting_center)[i].y)
				+ ((*new_clusting_center)[i].z - (*clusting_center)[i].z)*((*new_clusting_center)[i].z - (*clusting_center)[i].z);
		}
		std::cout << "literation done,L2= " << L2 << std::endl;
	} while (L2 >= L2_min);
	std::cout << "clusting done!" << std::endl;
}

template <> float SLIC<pcl::PointXYZRGB>::Calculate_SLIC_dist(const pcl::PointXYZRGB &cloud, float sp_dist)
{
	float c_dist = cloud.r*cloud.r + cloud.g*cloud.g + cloud.b*cloud.b;
	return std::sqrt(c_dist + m * m*sp_dist / (s*s));
}

template <> float SLIC<pcl::PointXYZ>::Calculate_SLIC_dist(const pcl::PointXYZ &cloud, float sp_dist)
{
	return std::sqrt(sp_dist)*m/s;
}


template<typename PointTT>
void SLIC<PointTT>::filterClustingSeed()
{
	typename pcl::UniformSampling<PointTT>::Ptr filter(new pcl::UniformSampling<PointTT>);
	typename pcl::PointCloud<PointTT>::Ptr seeds(new pcl::PointCloud<PointTT>);
	//std::cout << "filter start" << std::endl;
	filter->setInputCloud(cloud);
	filter->setRadiusSearch(s);
	filter->filter(*seeds);
	seed = seeds;
	std::cout << "number of seed: " << seed->width << std::endl;
}

template<typename PointTT>
void SLIC<PointTT>::getLabledCloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr &result)
{
	copyPointCloud(*cloud, *result);
	for (int i = result->width - 1; i >= 0; --i)
	{
		(*result)[i].label = label[i];
	}
}

template<typename PointTT>
void SLIC<PointTT>::getLabledCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &result)
{
	copyPointCloud(*cloud, *result);
	for (int i = result->width - 1; i >= 0; --i)
	{
		(*result)[i].label = label[i];
	}
}
template<typename PointTT>
void SLIC<PointTT>::getSeed(const typename pcl::PointCloud<PointTT>::Ptr &seed)
{
	pcl::copyPointCloud(*(this->seed), *seed);
}
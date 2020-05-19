#include<pcl/point_cloud.h>
#include<pcl/filters/random_sample.h>
#include<pcl/filters/uniform_sampling.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<vector>
#include<iostream>
#include<cmath>
#include<algorithm>
#include<random>
#include<ctime>
#include <numeric>

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
	void setM(float m){this->m = m;}
	void setS(float s){this->s = s;}
	void setL2_min(float L2_min){this->L2_min = L2_min;}
	void setInputCloud(pcl::PointCloud<PointTT> &cloud){this->cloud = cloud;}

	//socket out
	float getM() { return m; }
	float getS() { return s; }
	float geL2_min() { return L2_min; }

	void getSeed(const typename pcl::PointCloud<PointTT>::Ptr &seed);
	void getLabledCloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr &result);
	void getLabledCloud(const pcl::PointCloud<pcl::PointXYZRGBL>::Ptr &result);
	int  getSuperpixelCount() { return superpixelcount; }

	//actions
	void SLIC_superpointcloudclusting();

private:
	float m;
	float s;
	float L2_min;
	int superpixelcount;

	typename pcl::PointCloud<PointTT>::ConstPtr cloud;
	typename pcl::PointCloud<PointTT>::ConstPtr seed;
	std::vector<std::vector<int>> clusters;

	float Calculate_SLIC_dist(const PointTT &cloud, float sp_dist);

	void filterClustingSeed();

	std::vector<int> label;
};

template<typename PointTT>
float Cal_undersegmentation_error(const PointTT a, const PointTT gt, int classcount, int supercount,bool is_new=false);

typedef pcl::PointXYZ	PointT;
typedef pcl::PointXYZRGB	PointTC;
typedef pcl::PointXYZL		PoinTL;



int main(int argc, char* argv[])
{
	float s = 10.0f;
	float m = 1.0f;
	//int samples = 800;
	float L2_min = 10.0f;

	pcl::PointCloud<PoinTL>::Ptr cloud(new pcl::PointCloud<PoinTL>);
	pcl::io::loadPCDFile("C:\\Users\\37952\\Desktop\\000_orig_pcd\\benthi_control_A_D30_centre_filter.pcd", *cloud);

	pcl::PointCloud<PoinTL>::Ptr clusting_center(new pcl::PointCloud<PoinTL>);
	pcl::PointCloud<pcl::PointXYZL>::Ptr labledcloud(new pcl::PointCloud<pcl::PointXYZL>);

	SLIC<PoinTL> clusting(cloud,m,s,L2_min);
	clusting.SLIC_superpointcloudclusting();
	clusting.getLabledCloud(labledcloud);
	clusting.getSeed(clusting_center);
	
	float error = Cal_undersegmentation_error(labledcloud, cloud, 26, clusting.getSuperpixelCount(),true);

	std::cout << "Under-segmentation error is: " << error << std::endl;

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





template<typename PointTT>
void SLIC<PointTT>::SLIC_superpointcloudclusting()
{
	std::cout << "clusting start" << std::endl;
	float x, y, z, L2;
	float count = 0.0f;
	std::vector<float> slic_dist(cloud->width, 1000.0f);										//slic dist to be compared
	typename pcl::PointCloud<PointTT>::Ptr clusting_center(new pcl::PointCloud<PointTT>);		//center of each superpixel
	typename pcl::PointCloud<PointTT>::Ptr new_clusting_center(new pcl::PointCloud<PointTT>);	//to calculate L2 norm
	filterClustingSeed();
	pcl::copyPointCloud(*seed, *new_clusting_center);
	
	std::vector<int>search_indices;				//for K-nearst search
	std::vector<int>::iterator search_iter;

	std::vector<float>point_square_dist;
	std::vector<float>::iterator square_dist_iter;
	

	pcl::KdTreeFLANN<PointTT> kd_tree;
	kd_tree.setInputCloud(cloud);
	DWORD time,time_end;

	//SLIC step down here
	do
	{
		time = GetTickCount();
		label = std::vector<int>(cloud->width, 0);
		slic_dist = std::vector<float>(cloud->width, 1000.0f);
		pcl::copyPointCloud(*new_clusting_center, *clusting_center);
		
		for (int i = clusting_center->width - 1; i >= 0; i--)//each seed point
		{
			//find each point in 2S radius
			search_indices.clear();
			point_square_dist.clear();
			kd_tree.radiusSearch((*clusting_center)[i], 2 * s, search_indices, point_square_dist, 0);
			x = y = z = 0.0f; //clear x,y,z
			count = 0.0f;
			square_dist_iter = point_square_dist.begin();

			for (search_iter=search_indices.begin();search_iter!=search_indices.end(); search_iter++)//each point in 2S radius
			{
				float dist_slic = Calculate_SLIC_dist((*cloud)[*search_iter], *square_dist_iter);
				square_dist_iter++;

				if (dist_slic < slic_dist[*search_iter])
				{
					slic_dist[*search_iter] = dist_slic;
					label[*search_iter] = i;
					x += (*cloud)[*search_iter].x;
					y += (*cloud)[*search_iter].y;
					z += (*cloud)[*search_iter].z;
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
		//compute L2 norm and SLIC time
		time_end = GetTickCount();
		L2 = 0.0f;
		for (int i = clusting_center->width - 1; i >= 0; i--)
		{
			L2 += ((*new_clusting_center)[i].x - (*clusting_center)[i].x)*((*new_clusting_center)[i].x - (*clusting_center)[i].x)
				+ ((*new_clusting_center)[i].y - (*clusting_center)[i].y)*((*new_clusting_center)[i].y - (*clusting_center)[i].y)
				+ ((*new_clusting_center)[i].z - (*clusting_center)[i].z)*((*new_clusting_center)[i].z - (*clusting_center)[i].z);
		}
		std::cout << "literation done,L2= " << L2 <<"!! using time: "<<int(time_end-time)<<"ms"<< std::endl;		
	} while (L2 >= L2_min);
	seed = new_clusting_center;
	//just to show that its done
	std::cout << "clusting done!" << std::endl;

}

template <> float SLIC<pcl::PointXYZRGB>::Calculate_SLIC_dist(const pcl::PointXYZRGB &cloud, float sp_dist)
{
	float c_dist = cloud.r*cloud.r + cloud.g*cloud.g + cloud.b*cloud.b;
	return std::sqrt(c_dist + m * m*sp_dist / (s*s));
}

template <> float SLIC<pcl::PointXYZRGBL>::Calculate_SLIC_dist(const pcl::PointXYZRGBL &cloud, float sp_dist)
{
	float c_dist = cloud.r*cloud.r + cloud.g*cloud.g + cloud.b*cloud.b;
	return std::sqrt(c_dist + m * m*sp_dist / (s*s));
}

//if there is some difference with m or without m?
template <> float SLIC<pcl::PointXYZ>::Calculate_SLIC_dist(const pcl::PointXYZ &cloud, float sp_dist)
{
	return std::sqrt(sp_dist)*m/s;
}

template <> float SLIC<pcl::PointXYZL>::Calculate_SLIC_dist(const pcl::PointXYZL &cloud, float sp_dist)
{
	return std::sqrt(sp_dist)*m / s;
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
	superpixelcount = seed->width;
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

template<typename PointTT>
float Cal_undersegmentation_error(const PointTT a, const PointTT gt,const int classcount,const int supercount,bool is_new)
{
	//init hashmap
	std::vector < std::vector<int> > hashmap(classcount, std::vector<int>());
	std::vector<int> label_count(classcount,0);
	std::vector<int> pixel_count(supercount, 0);
	std::vector<int> label_pixel(classcount, 0);
	std::vector<std::vector<int>>::iterator lit1;
	std::vector<int>::iterator lit2;
	int errcount = 0;

	for (int i = classcount - 1; i >= 0; --i)
	{
		hashmap[i].resize(supercount,false);
	}

	//find superpixel that covers the correspond label
	//count the number of points corresponds to each label
	for (int i = gt->width - 1; i >= 0; i--)
	{
		hashmap[(*gt)[i].label][(*a)[i].label] +=1;
		label_count[(*gt)[i].label] += 1;
		pixel_count[(*a )[i].label] += 1;

	}
	
	//count superpixels
	if (is_new)
	{
		for (lit1 = hashmap.begin(); lit1 != hashmap.end(); lit1++)
		{
			for (lit2 = lit1->begin(); lit2 != lit1->end(); lit2++)
			{
				//add smaller part(orig or outline) into count
				if ((*lit2) != 0) label_pixel[distance(hashmap.begin(), lit1)] += std::min(pixel_count[distance(lit1->begin(), lit2)]-*lit2,*lit2);
			}
		}
		errcount = std::accumulate(label_pixel.begin(), label_pixel.end(), 0);
	}
	else
	{
		for (lit1 = hashmap.begin(); lit1 != hashmap.end(); lit1++)
		{
			for (lit2 = lit1->begin(); lit2 != lit1->end(); lit2++)
			{
				//add all pixel into count
				if ((*lit2) != 0) label_pixel[distance(hashmap.begin(), lit1)] += pixel_count[distance(lit1->begin(), lit2)];
			}
		}
		errcount = std::accumulate(label_pixel.begin(), label_pixel.end(), 0) - gt->width;
	}
	return float(errcount) / float(gt->width);
}

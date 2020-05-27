#ifndef _SLIC_
#define _SLIC_
#include<pcl/point_cloud.h>
#include<pcl/filters/uniform_sampling.h>
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
	void setM(float m) { this->m = m; }
	void setS(float s) { this->s = s; }
	void setL2_min(float L2_min) { this->L2_min = L2_min; }
	void setInputCloud(pcl::PointCloud<PointTT> &cloud) { this->cloud = cloud; }

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

	std::vector<int>search_indices;				//for K-means search
	std::vector<int>::iterator search_iter;

	std::vector<float>point_square_dist;
	std::vector<float>::iterator square_dist_iter;


	pcl::KdTreeFLANN<PointTT> kd_tree;
	kd_tree.setInputCloud(cloud);
	PointTT pointsearch;				//K nearst search used by new center finding
	std::vector<int> search(1, 0);
	std::vector<float>searchdist(1, 0);	//never mind
	DWORD time, time_end;

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

			for (search_iter = search_indices.begin(); search_iter != search_indices.end(); search_iter++)//each point in 2S radius
			{
				float dist_slic = Calculate_SLIC_dist((*cloud)[*search_iter], *square_dist_iter);
				square_dist_iter++;

				if (dist_slic < slic_dist[*search_iter])
				{
					slic_dist[*search_iter] = dist_slic;
					label[*search_iter] = i;
					x += cloud->points[*search_iter].x;
					y += cloud->points[*search_iter].y;
					z += cloud->points[*search_iter].z;
					count += 1.0f;
				}
			}
			//compute new seed
			if (count != 0)
			{
				pointsearch.x = x / count;
				pointsearch.y = y / count;
				pointsearch.z = z / count;
				kd_tree.nearestKSearch(pointsearch, 1, search, searchdist);				//find new point in center of the superpixel
				new_clusting_center->points[i].x = cloud->points[search[0]].x;
				new_clusting_center->points[i].y = cloud->points[search[0]].y;
				new_clusting_center->points[i].z = cloud->points[search[0]].z;
			}
		}
		//compute L2 norm and SLIC time
		time_end = GetTickCount();
		L2 = 0.0f;
		for (int i = clusting_center->width - 1; i >= 0; i--)
		{
			L2 += (new_clusting_center->points[i].x - clusting_center->points[i].x)*(new_clusting_center->points[i].x - clusting_center->points[i].x)
				+ (new_clusting_center->points[i].y - clusting_center->points[i].y)*(new_clusting_center->points[i].y - clusting_center->points[i].y)
				+ (new_clusting_center->points[i].z - clusting_center->points[i].z)*(new_clusting_center->points[i].z - clusting_center->points[i].z);
		}
		std::cout << "literation done,L2= " << L2 << "!! using time: " << int(time_end - time) << "ms" << std::endl;
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
	return std::sqrt(sp_dist)*m / s;
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


#endif // !_SLIC


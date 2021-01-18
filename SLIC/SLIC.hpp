#ifndef _SLIC_
#define _SLIC_
#include<pcl/point_cloud.h>
#include<pcl/filters/uniform_sampling.h>
#include<pcl/filters/random_sample.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<vector>
#include<iostream>
#include<cmath>
#include<random>
#include<functional>
#include<algorithm>

using namespace std;
using namespace pcl;


template<typename PointTT>
typename PointCloud<PointTT>::Ptr UniformSampling(typename PointCloud<PointTT>::Ptr const cloud,float const s);

template <typename PointTT>
typename PointCloud<PointTT>::Ptr RandomSampling(typename PointCloud<PointTT>::Ptr const cloud,float const nsample);

template<typename PointTT>
class SLIC
{
public:
	//	constructor/deconstructor
	SLIC() {};
	SLIC(const typename PointCloud<PointTT>::Ptr &cloud, float m = 1.0f, float s = 1.0f, float L2_min = 1.0f) :m(m), s(s), L2_min(L2_min)
	{
		this->cloud = cloud;
	};
	~SLIC() {};

	//socket in
	void setM(float m) { this->m = m; }
	void setS(float s) { this->s = s; }
	void setL2_min(float L2_min) { this->L2_min = L2_min; }
	void setInputCloud(PointCloud<PointTT> &cloud) { this->cloud = cloud; }
	
//	template<typename F>
//	void setSamplingFunc(F& func){sampling = func;}

	//socket out
	float getM() { return m; }
	float getS() { return s; }
	float geL2_min() { return L2_min; }

	void getSeed(const typename PointCloud<PointTT>::Ptr &seed);
	void getLabledCloud(const PointCloud<PointXYZL>::Ptr &result);
	void getLabledCloud(const PointCloud<PointXYZRGBL>::Ptr &result);
	int  getSuperpixelCount() { return superpixelcount; }

	//actions
	void SLIC_superpointcloudclusting();

private:
	float m;
	float s;
	float L2_min;
	int superpixelcount;

	typename PointCloud<PointTT>::Ptr cloud;
	typename PointCloud<PointTT>::Ptr seed;
	vector<vector<int>> clusters;
//	function<typename PointCloud<PointTT>::Ptr(typename PointCloud<PointTT>::Ptr const)> sampling;
	//dsit function,defaults to XYZ
	float Calculate_SLIC_dist(const PointTT& cloud,const PointTT& seed, float sp_dist) { return sqrt(sp_dist) * m / s; };


	vector<int> label;
};


template<typename PointTT>
void SLIC<PointTT>::SLIC_superpointcloudclusting()
{
	cout << "clusting start" << endl;
	float x, y, z, L2;
	float count = 0.0f;
	vector<float> slic_dist(cloud->width, 1000.0f);										//slic dist to be compared
	typename PointCloud<PointTT>::Ptr clusting_center(new PointCloud<PointTT>);		//center of each superpixel
	typename PointCloud<PointTT>::Ptr new_clusting_center(new PointCloud<PointTT>);	//to calculate L2 norm
	this->seed = RandomSampling<PointTT>(this->cloud, 50);
	copyPointCloud(*seed, *new_clusting_center);

	vector<int>search_indices;				//for K-means search

	vector<float>point_square_dist;
	vector<float>::iterator square_dist_iter;


	KdTreeFLANN<PointTT> kd_tree;
	kd_tree.setInputCloud(cloud);
	PointTT pointsearch;				//K nearst search used by new center finding
	vector<int> search(1, 0);
	vector<float>searchdist(1, 0);	//never mind
	DWORD time, time_end;

	//SLIC step down here
	size_t iter = 0;
	do
	{
		//time = GetTickCount();
		label = vector<int>(cloud->width, 0);
		slic_dist = vector<float>(cloud->width, 1000.0f);
		copyPointCloud(*new_clusting_center, *clusting_center);
		vector<float> counter(seed->width, 0.0);
		//too much for!!!
		for (int i = clusting_center->width - 1; i >= 0; i--)//for each seed point we compute a hard association using pointed dist function
		{
			//find each point in 2S radius
			search_indices.clear();
			point_square_dist.clear();
			kd_tree.radiusSearch((*clusting_center)[i], 2.0d * double(s), search_indices, point_square_dist, 0);//find nearst indices using knn and return the spatical dist
			x = y = z = 0.0f; //clear x,y,z
			count = 0.0f;
			square_dist_iter = point_square_dist.begin();

			for (auto search_iter:search_indices)//each point in 2S radius
			{
				float dist_slic = Calculate_SLIC_dist((*cloud)[search_iter],clusting_center->points[i], *square_dist_iter);//cal the slic dist and add  doing the k-means process
				square_dist_iter++;

				if (dist_slic < slic_dist[search_iter])
				{
					slic_dist[search_iter] = dist_slic;
					label[search_iter] = i;
				}
			}
		}
		for (auto center : new_clusting_center->points)//clear clusting center
		{
			center.x = center.y = center.z = 0;//clear center count
		}
		//update spix center
		//cal the center of the spix ,then find the nearst point in the cloud to replace it
		for (auto i = 0; i < label.size(); i++)
		{	
			new_clusting_center->points[label[i]].x += cloud->points[i].x;
			new_clusting_center->points[label[i]].y += cloud->points[i].y;
			new_clusting_center->points[label[i]].z += cloud->points[i].z;
			counter[label[i]] += 1.0;
		}
		for (auto i = 0; i < new_clusting_center->width; i++)
		{
			vector<int> idx(1);
			vector<float> idxx(1);
			if (0.0 != counter[i])
			{
				new_clusting_center->points[i].x /= counter[i];
				new_clusting_center->points[i].y /= counter[i];
				new_clusting_center->points[i].z /= counter[i];
			}
			kd_tree.nearestKSearch(new_clusting_center->points[i], 1, idx, idxx);
			memcpy(&(new_clusting_center->points[i]), &(cloud->points[idx[0]]), sizeof((new_clusting_center->points[i])));
		}

		//compute L2 norm and SLIC time
		//time_end = GetTickCount();
		L2 = 0.0f;
		for (int i = clusting_center->width - 1; i >= 0; i--)
		{
			L2 += (new_clusting_center->points[i].x - clusting_center->points[i].x)*(new_clusting_center->points[i].x - clusting_center->points[i].x)
				+ (new_clusting_center->points[i].y - clusting_center->points[i].y)*(new_clusting_center->points[i].y - clusting_center->points[i].y)
				+ (new_clusting_center->points[i].z - clusting_center->points[i].z)*(new_clusting_center->points[i].z - clusting_center->points[i].z);
		}
		//cout << "literation done,L2= " << L2 << "!! using time: " << int(time_end - time) << "ms" << endl;
		iter++;
	} while (L2 >= L2_min || iter<10);
	cout << iter << endl;
	seed = new_clusting_center;
	//just to show that its done
	cout << "clusting done!" << endl;

}


template<typename PointTT>
typename PointCloud<PointTT>::Ptr UniformSampling(typename PointCloud<PointTT>::Ptr const cloud, float const s)
{
	typename UniformSampling<PointTT>::Ptr filter(new UniformSampling<PointTT>);
	typename PointCloud<PointTT>::Ptr seeds(new PointCloud<PointTT>);
	//cout << "filter start" << endl;
	filter->setInputCloud(cloud);
	filter->setRadiusSearch(s);
	filter->filter(*seeds);
	return seeds;
}

template <typename PointTT>
typename PointCloud<PointTT>::Ptr RandomSampling(typename PointCloud<PointTT>::Ptr const cloud,float const nsample)
{
	RandomSample<PointTT> filter;
	filter.setInputCloud(cloud);
	filter.setSample(nsample);
	typename PointCloud<PointTT>::Ptr ret(new PointCloud<PointTT>);
	filter.filter(*ret);
	return ret;
}


//template <> float SLIC<PointT>::Calculate_SLIC_dist(const PointTT& cloud,const pointTT& seed, float sp_dist) { return 0; };
// partitial specialization Calculate_SLIC_dist in class SLIC to implement an dist function for a given pointcloud type

template <> float SLIC<PointXYZRGB>::Calculate_SLIC_dist(const PointXYZRGB &cloud, const PointXYZRGB& seed, float sp_dist)
{
	float c_dist = pow(cloud.r-seed.r,2) + pow(cloud.g-seed.g,2) + pow(cloud.b-seed.b,2);
	return sqrt(c_dist + m * m*sp_dist / (s*s));
}

template <> float SLIC<PointXYZRGBL>::Calculate_SLIC_dist(const PointXYZRGBL &cloud, const PointXYZRGBL& seed, float sp_dist)
{
	float c_dist = pow(cloud.r - seed.r, 2) + pow(cloud.g - seed.g, 2) + pow(cloud.b - seed.b, 2);
	return sqrt(c_dist + m * m*sp_dist / (s*s));
}

template <> float SLIC<PointNormal>::Calculate_SLIC_dist(const PointNormal& cloud,const PointNormal& seed,float sp_dist)
{
	float N_dist = 1 - pow(cloud.normal_x * seed.normal_x + cloud.normal_y * seed.normal_y + cloud.normal_z * seed.normal_z,2);//sin^2
	return sqrt(sp_dist + pow(m/s,2)*N_dist);
}

//using curvature instead of normal to calculate the dist,intensity is useless 
template <> float SLIC<PointXYZINormal>::Calculate_SLIC_dist(const PointXYZINormal& cloud, const PointXYZINormal& seed, float sp_dist)
{
	float N_dist = pow(abs(cloud.curvature-seed.curvature),2);
	return sqrt(sp_dist + pow(m / s, 2) * N_dist);
}

template<typename PointTT>
void SLIC<PointTT>::getLabledCloud(const PointCloud<PointXYZL>::Ptr &result)
{
	copyPointCloud(*cloud, *result);
	for (int i = result->width - 1; i >= 0; --i)
	{
		(*result)[i].label = label[i];
	}
}

template<typename PointTT>
void SLIC<PointTT>::getLabledCloud(const PointCloud<PointXYZRGBL>::Ptr &result)
{
	copyPointCloud(*cloud, *result);
	for (int i = result->width - 1; i >= 0; --i)
	{
		(*result)[i].label = label[i];
	}
}
template<typename PointTT>
void SLIC<PointTT>::getSeed(const typename PointCloud<PointTT>::Ptr &seed)
{
	copyPointCloud(*(this->seed), *seed);
}


#endif // !_SLIC_


#include<pcl/point_cloud.h>
#include<pcl/filters/random_sample.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<pcl/kdtree/kdtree_flann.h>
#include<vector>
#include<iostream>
#include<cmath>
#include<random>

using namespace pcl;

std::vector<unsigned char> R = { 
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240 };
std::vector<unsigned char> G = { 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 };
std::vector<unsigned char> B = { 
	0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

std::vector<int> redrict(65);

typedef PointXYZ PointT;
typedef PointXYZRGB PointTC;

int main()
{	
	float s = 0.2f;
	float m = 1.0f;
	int samples = 200;
	float L2_min = 0.05f;
/*
	for (int i = 0; i < 13; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			redrict[i * 13 + j] = j * 13 + i;
			std::cout << redrict[i * 13 + j] << ' ';
		}
	}
	*/
	
	
	//load new point cloud
	PointCloud<PointT>::Ptr cloud(new PointCloud<PointT>);
	io::loadPCDFile("C:\\Users\\37952\\Desktop\\desk.pcd", *cloud);

	//sampling seeds
	RandomSample<PointT>::Ptr rdm(new RandomSample<PointT>);
	rdm->setInputCloud(cloud);
	rdm->setSample(samples);
	PointCloud<PointT>::Ptr clusteing_center(new PointCloud<PointT>);
	PointCloud<PointT>::Ptr new_clusteing_center(new PointCloud<PointT>);
	rdm->filter(*clusteing_center);
	copyPointCloud(*clusteing_center, *new_clusteing_center);



	std::vector<int> lable(cloud->width);
	std::vector<float> slic_dist(cloud->width, 1000.0f);

	//sampling 
	KdTreeFLANN<PointT> kd_tree;
	kd_tree.setInputCloud(cloud);
	std::vector<int>search_indices;
	std::vector<float>point_square_dist;
	float x, y, z,L2;
	do
	{
		copyPointCloud(*new_clusteing_center, *clusteing_center);
		for (int i = clusteing_center->width - 1; i >= 0; i--)//each seed point
		{
			search_indices.clear();
			point_square_dist.clear();
			kd_tree.radiusSearch((*clusteing_center)[i], 2 * s, search_indices, point_square_dist, 0);
			x = y = z = 0; //clear x,y,z
			float count = 0.0f;
			for (int j = search_indices.size() - 1; j >= 0; j--)//each point in 2S radius
			{
				float dist_color = 0.0f;
				float dist_slic = std::sqrt(dist_color + m * m*point_square_dist[j] / (s*s));
				if (dist_slic < slic_dist[search_indices[j]])
				{
					slic_dist[search_indices[j]] = dist_slic;
					lable[search_indices[j]] = i;
					x += (*cloud)[search_indices[j]].x;
					y += (*cloud)[search_indices[j]].y;
					z += (*cloud)[search_indices[j]].z;
					count += 1.0f;
				}
			}
			//compute new seed
			(*new_clusteing_center)[i].x = x / count;
			(*new_clusteing_center)[i].y = y / count;
			(*new_clusteing_center)[i].z = z / count;
		}
		//compute L2 norm
		L2 = 0.0f;
		for (int i = clusteing_center->width - 1; i >= 0; --i)
		{
			L2 +=	((*new_clusteing_center)[i].x - (*clusteing_center)[i].x)*((*new_clusteing_center)[i].x - (*clusteing_center)[i].x)
				+	((*new_clusteing_center)[i].y - (*clusteing_center)[i].y)*((*new_clusteing_center)[i].y - (*clusteing_center)[i].y)
				+	((*new_clusteing_center)[i].z - (*clusteing_center)[i].z)*((*new_clusteing_center)[i].z - (*clusteing_center)[i].z);
		}
		std::cout << "literation done L2= "<<L2 << endl;
	} while (L2 >= L2_min);

	PointCloud<PointTC>::Ptr labledcloud(new PointCloud<PointTC>);
	copyPointCloud(*cloud, *labledcloud);
	for (int i = labledcloud->width - 1; i >= 0; --i)
	{
		
		(*labledcloud)[i].r = R[lable[i] * 65 / samples];
		(*labledcloud)[i].g = G[lable[i] * 65 / samples];
		(*labledcloud)[i].b = B[lable[i] * 65 / samples];
	}


	std::cout << "done" << std::endl;
/*
	for (int i = lable.size() - 1; i >= 0; --i)
	{
		cout << (*labledcloud)[i].label << '	';
		if (!(i % 15)) std::cout << std::endl;
	}
	*/
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("SLIC"));
	int v1(0), v2(1), v3(2);
	viewer->createViewPort(0, 0, 0.33, 1, v1);
	viewer->addPointCloud<PointT>(cloud, "origional cloud", v1);
	viewer->createViewPort(0.33, 0, 66, 1, v2);
	viewer->addPointCloud<PointTC>(labledcloud,"labled cloud", v2);
	viewer->createViewPort(0.66, 0, 1, 1, v3);
	viewer->addPointCloud<PointT>(clusteing_center, "new cloud", v3);
	viewer->spin();

	return 0;
}
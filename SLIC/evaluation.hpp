#pragma once
#include<ctime>
#include <numeric>
#include<vector>
#include<cmath>
#include<pcl/point_cloud.h>
#include<algorithm>

#ifndef _POINTPIXEL_VAL
#define _POINTPIXEL_VAL
template<typename PointTT>

float Cal_undersegmentation_error(const PointTT a, const PointTT gt, const int classcount, const int supercount, bool is_new)
{
	//init hashmap
	std::vector < std::vector<int> > hashmap(classcount, std::vector<int>());
	std::vector<int> label_count(classcount, 0);
	std::vector<int> pixel_count(supercount, 0);
	std::vector<int> label_pixel(classcount, 0);
	std::vector<std::vector<int>>::iterator lit1;
	std::vector<int>::iterator lit2;
	int errcount = 0;

	for (int i = classcount - 1; i >= 0; --i)
	{
		hashmap[i].resize(supercount, 0);
	}

	//find superpixel that covers the correspond label
	//count the number of points corresponds to each label
	for (int i = gt->width - 1; i >= 0; i--)
	{
		hashmap[(*gt)[i].label][(*a)[i].label] += 1;
		label_count[(*gt)[i].label] += 1;
		pixel_count[(*a)[i].label] += 1;

	}

	//count superpixels
	if (is_new)
	{
		for (lit1 = hashmap.begin(); lit1 != hashmap.end(); lit1++)
		{
			for (lit2 = lit1->begin(); lit2 != lit1->end(); lit2++)
			{
				//add smaller part(orig or outline) into count
				if ((*lit2) != 0) label_pixel[distance(hashmap.begin(), lit1)] += std::min(pixel_count[distance(lit1->begin(), lit2)] - *lit2, *lit2);
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


template<typename PointTT>
float Cal_Achievable_seg_acc(const PointTT a, const PointTT gt, const int classcount, const int supercount)
{
	std::vector<std::vector<int>> hashmap(supercount, std::vector<int>(classcount, 0));
	std::vector<std::vector<int>>::iterator lit;
	std::vector<int> classpred(classcount, 0);
	std::vector<int> classgt(classcount, 0);

	for (int i = gt->width - 1; i >= 0; --i)
	{
		hashmap[(*a)[i].label][(*gt)[i].label] += 1;
		classgt[(*gt)[i].label] += 1;
	}

	for (lit = hashmap.begin(); lit != hashmap.end(); lit++)
	{
		auto maxpos = std::max_element(lit->begin(), lit->end());
		classpred[maxpos - lit->begin()] += *maxpos;
	}

	return float(std::accumulate(classpred.begin(), classpred.end(), 0)) / float(std::accumulate(classgt.begin(), classgt.end(), 0));
}

#endif // !_POINTPIXEL_VAL




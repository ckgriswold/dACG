#include<new>
#include<math.h>
#include<iostream>
#include<vector>

#include"header.h"
#include"Mersenne.h"

void make_prelim_graph(std::vector< float* > *tree, int sample_size, float sigma, int num_intr, float t_max, 
						int *num_coal, int *num_sel, int *num_other,
						int block_size, std::vector< int* > *block_start_stop) {
	
	float t = 0.0;
	float total_rate = 0.0;
	float ran_dev = 0.0;
	
	int k = sample_size;
	int node_count_for_block = 0;
	int node_count_tot = 0;
	
	int block_start = 0;
	
	for(int i = 0; i <= sample_size - 1; i++) {
		float *graph_info;
		graph_info = new float[2];
		graph_info[0] = t;
		graph_info[1] = 0.0;
		tree->push_back(graph_info);
		
		node_count_tot = node_count_tot + 1;
		node_count_for_block = node_count_for_block + 1;
	}
	
	*num_coal = 0;
	*num_sel = 0;
	*num_other = 0;
	
	while(t < t_max && k > 1)	{													
		ran_dev = genrand_real3();
		
		total_rate = float(k * (k - 1)/2) + float(k) * sigma/2;
		t += -1.0 * log(1.0 -  genrand_real3())/total_rate;
																		
		if(ran_dev <= float(k * (k - 1)/2)/total_rate)	{
			/* Coalescent event */
			float *graph_info;
			graph_info = new float[2];
			graph_info[0] = t;
			graph_info[1] = 1.0;
			tree->push_back(graph_info);
			
			k = k - 1;
			*num_coal = *num_coal + 1;
			
			node_count_tot = node_count_tot + 1;
			node_count_for_block = node_count_for_block + 1;
		}
		
		else if((ran_dev > float(k * (k - 1)/2)/total_rate) && (ran_dev <= 1.0))	{
			/* Selection Event */
			float *graph_info;
			
			graph_info = new float[2];
			graph_info[0] = t;
			graph_info[1] = 2.0;
			tree->push_back(graph_info);
			
			graph_info = new float[2];
			graph_info[0] = t;
			graph_info[1] = 2.0;
			tree->push_back(graph_info);
			
			for(int i = 0; i <= num_intr - 1; i++)	{
				graph_info = new float[2];
				graph_info[0] = t;
				graph_info[1] = 2.0;
				tree->push_back(graph_info);
			}
			
			k = k + 1 + num_intr;
			*num_sel = *num_sel + 1;
			
			node_count_tot = node_count_tot + 2 + num_intr;
			node_count_for_block = node_count_for_block + 2 + num_intr;
		}
		
		else	{
			std::cout << "Error determining if a coal or selection event occurred." << std::endl;
			exit(8);
		}
		
		//std::cout << node_count_for_block << " " << k << " " << t << std::endl;
		
		if((node_count_for_block >= block_size && node_count_for_block >= k) ||
				t >= t_max || k == 1)	{
			/* Graph Slice */
			
			//std::cout << node_count_for_block << " " << k << " " << t << std::endl;
			
			int *block_info;
			block_info = new int[2];
			
			block_info[0] = block_start;
			block_start = node_count_tot;
			
			for(int i = 0; i <= k - 1; i++)	{
				float *graph_info;
			
				graph_info = new float[2];
				graph_info[0] = t;
				graph_info[1] = 3.0;
				tree->push_back(graph_info);
				
				node_count_tot = node_count_tot + 1;
				*num_other = *num_other + 1;
			}
			
			block_info[1] = node_count_tot;
			block_start_stop->push_back(block_info);

			node_count_for_block = 0;
			
			//std::cout << block_info[0] << " " << block_info[1] << std::endl;
		}
		
		//std::cout << node_count_for_block << " " << k << " " << t << std::endl;
		
	}
	
	std::cout << "Exiting make_prelim_graph" << std::endl;
	return;
}
			
		
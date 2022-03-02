#include<stdlib.h>
#include<stdio.h>
#include<new>
#include<vector>
#include<time.h>
#include<iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include"header.h"

int gpu_graph(int sample_size, float sigma, int num_intr, float t_max, float x_mu, int num_dim, float g_mu, int num_reg,
				int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores,
				int graph_cycles_mult, int block_size,
				float a_1, float a_2, float a_3, float alpha, float K,
				int *&desc1, int *&desc2, int *&ancestor1, int *&ancestor2, int *&intr_ind,
				float *&x, int *&genotype_1, int *&genotype_2,
				
				cl_context context_calc_x, cl_command_queue command_queue_calc_x, 
				cl_program program_calc_x, cl_kernel kernel_calc_x,
				
				cl_context context_rnd_uniform, cl_command_queue command_queue_rnd_uniform,
  				cl_program program_rnd_uniform, cl_kernel kernel_rnd_uniform,
  				
  				cl_context context_mut, cl_command_queue command_queue_mut,
  				cl_program program_mut, cl_kernel kernel_mut,
  				
  				cl_context context_calc_k, cl_command_queue command_queue_calc_k,
  				cl_program program_calc_k, cl_kernel kernel_calc_k,
  				
  				cl_context context_time_to_anc, cl_command_queue command_queue_time_to_anc,
  				cl_program program_time_to_anc, cl_kernel kernel_time_to_anc,
  				
  				cl_context context_calc_delta_x, cl_command_queue command_queue_calc_delta_x,
  				cl_program program_calc_delta_x, cl_kernel kernel_calc_delta_x
				
				) {
		
		/* Setup prelim_graph data object, which is a c++ vector of pointers to floats 								*/
		/* Each element in the vector will be a 2-dimensional array that consisting of the time of and event in the */
		/* graph and the type of event that occurred.																*/
		std::vector< float* > *prelim_graph;
		prelim_graph = new std::vector< float* >;
		
		/* Setup a c++ vector that stores the beginning and end positions of slices of the ancestral graph */
		std::vector< int* > *block_start_stop;
		block_start_stop = new std::vector< int* >;
		
		int num_coal;
		int num_sel;
		int num_other;
		
		/* The make_prelim_graph function generates information in the prelim_graph data object 					*/
		make_prelim_graph(prelim_graph, sample_size, sigma, num_intr, t_max, &num_coal, &num_sel, &num_other,
							block_size, block_start_stop);		
		
		/* A check for whether the graph and its contents will be too big for the memory capacity of a computer.  	*/					
		if(prelim_graph->size() * num_reg >  2147483647)	{
			std::cout << "The combination of graph size and number of genomic regions is too large." << std::endl;
			std::cout << "Reduce the rate of selection, number of interacting individuals or number of genomic regions." << std::endl;
			exit(8);
		}
		
		/* Set up data objects for ancestral graph, which is represented by a set of vectors.						*/
		desc1 = new int[prelim_graph->size()];
		desc2 = new int[prelim_graph->size()];
		ancestor1 = new int[prelim_graph->size()];
		ancestor2 = new int[prelim_graph->size()];
		intr_ind = new int[prelim_graph->size() * num_intr];
		
		x = new float[prelim_graph->size() * num_dim];
		
		int *x_updated = new int[prelim_graph->size()];
		
		genotype_1 = new int[prelim_graph->size() * num_reg];
		genotype_2 = new int[prelim_graph->size() * num_reg];
		
		for(int i = 0; i <= prelim_graph->size() * num_reg - 1; i++)	{
			genotype_1[i] = 0;
			genotype_2[i] = 0;
		}
	
		float *rand_uniform_1;
		int tot_num_events;
		tot_num_events = 2 * num_coal + num_sel + num_other;
		
		/* Here, a GPU kernel will be used for the first time.  It generates a large vector of uniform random deviates	*/
		/* that will be used in the connect_graph function.																*/
		calc_rand_uniform_cl(tot_num_events, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);	
								
		std:: cout << tot_num_events << std::endl;

		float *prelim_graph_node_event = new float[prelim_graph->size()];
		for(int i = 0; i <= prelim_graph->size() - 1; i++) prelim_graph_node_event[i] = prelim_graph->at(i)[1];
		
		/* The connect_graph function takes the prelim_graph information about times of an event and the type of event	*/
		/* and then generates the graph with nodes connected by edges.													*/
		connect_graph(prelim_graph_node_event, prelim_graph->size(), num_dim, num_intr, desc1, desc2, ancestor1, ancestor2,
						intr_ind, x_updated, x, rand_uniform_1);
						
		delete[] rand_uniform_1;
		
		/* Now we work from the bottom of the graph upwards in sections to determine the state of nodes.  This is 		*/
		/* done in sections of the graph defined by the block_start_stop vector that was previously generated.			*/
		for(int blk = block_start_stop->size() - 1; blk >= 0; blk--)	{
			block_size = block_start_stop->at(blk)[1] - block_start_stop->at(blk)[0];
			
			int block_start = block_start_stop->at(blk)[0];
			int block_stop = block_start_stop->at(blk)[1];
			
			std::cout << block_start << " " << block_stop << " " << prelim_graph->size() << std::endl;
			
			float *prelim_graph_node_time = new float[block_size];
	
			for(int i = 0; i <= block_size - 1; i++)	
				prelim_graph_node_time[i] = prelim_graph->at(i + block_start)[0];
			
			/* Get times along edges that connect nodes using the GPU kernel time_to_anc.cl							*/
			float *time_to_anc_1;
			float *time_to_anc_2;
			
			int *block_ancestor = new int[block_size];
			for(int i = 0; i <= block_size - 1; i++)	
				block_ancestor[i] = ancestor1[i + block_start] - block_start;
		
			time_to_anc_cl(prelim_graph_node_time, block_size, block_start, block_ancestor, &time_to_anc_1,
								context_time_to_anc, command_queue_time_to_anc, program_time_to_anc, kernel_time_to_anc,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
								
			delete[] block_ancestor;
			block_ancestor = new int[block_size];
			for(int i = 0; i <= block_size - 1; i++)	
				block_ancestor[i] = ancestor2[i + block_start] - block_start;
							
			time_to_anc_cl(prelim_graph_node_time, block_size, block_start, block_ancestor, &time_to_anc_2,
								context_time_to_anc, command_queue_time_to_anc, program_time_to_anc, kernel_time_to_anc,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			/* Next, generate two arrays of uniform deviates that will then be used to generate the number of mutations		*/
			/* along an edge in the graph.  This uses the MWC64X.cl kernel	of Thomas called through the 					*/
			/* calc_rand_uniform_cl function																				*/
			float *rand_uniform_2;
		
			calc_rand_uniform_cl(block_size, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_size, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			/* Using the uniform deviates.  The GPU is then used to generate the number of mutations along an edge that		*/
			/* affect phenotype.																							*/
			int *k_1;
			int *k_2;
		
			calc_k_cl(time_to_anc_1, block_size, &x_mu, rand_uniform_1, rand_uniform_2, &k_1,
							context_calc_k, command_queue_calc_k,program_calc_k,kernel_calc_k,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
								
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
			
			calc_rand_uniform_cl(block_size, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_size, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			calc_k_cl(time_to_anc_2, block_size, &x_mu, rand_uniform_1, rand_uniform_2, &k_2,
							context_calc_k, command_queue_calc_k,program_calc_k,kernel_calc_k,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			/* A new set of random uniform deviates is generated to then be used to generate normal deviates that correspond  	*/
			/* to mutations affecting phenoype.																					*/
			float *delta_x_1;
			float *delta_x_2;
		
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
		
			calc_rand_uniform_cl(block_size * num_dim, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_size * num_dim, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			/* Here the GPU is generates the normal deviates affecting phenotype.												*/
			calc_delta_x_cl(k_1, num_dim, rand_uniform_1, rand_uniform_2, block_size, &delta_x_1,
							context_calc_delta_x, command_queue_calc_delta_x, program_calc_delta_x, kernel_calc_delta_x,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
								
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
							
			calc_rand_uniform_cl(block_size * num_dim, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_size * num_dim, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
								
			calc_delta_x_cl(k_2, num_dim, rand_uniform_1, rand_uniform_2, block_size, &delta_x_2,
							context_calc_delta_x, command_queue_calc_delta_x, program_calc_delta_x, kernel_calc_delta_x,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			delete[] k_1;
			delete[] k_2;
			
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
			
			/* Next, uniform deviates and the number of mutations affecting the marker genotype are generated on the GPU	*/				
			calc_rand_uniform_cl(block_size, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_size, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
		
			calc_k_cl(time_to_anc_1, block_size, &g_mu, rand_uniform_1, rand_uniform_2, &k_1,
							context_calc_k, command_queue_calc_k,program_calc_k,kernel_calc_k,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
								
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
			
			calc_rand_uniform_cl(block_size, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_size, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);					
								
			calc_k_cl(time_to_anc_2, block_size, &g_mu, rand_uniform_1, rand_uniform_2, &k_2,
							context_calc_k, command_queue_calc_k,program_calc_k,kernel_calc_k,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);						
			
			int block_tot_k_1 = 0;
			int block_tot_k_2 = 0;
		
			int *block_sum_k_1 = new int[block_size];
			int *block_sum_k_2 = new int[block_size];
			
			for(int i = 0; i <= block_size - 1; i++) {
				block_sum_k_1[i] = block_tot_k_1;
				block_tot_k_1 = block_tot_k_1 + k_1[i];
				
				block_sum_k_2[i] = block_tot_k_2;
				block_tot_k_2 = block_tot_k_2 + k_2[i];
			}	
			
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
		
			calc_rand_uniform_cl(block_tot_k_1, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_rand_uniform_cl(block_tot_k_2, &rand_uniform_2, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			/* The sequenced region is divided into regions, each consisting of 15 sites.  The total length of the 				*/
			/* genomic region that is sequenced is num_reg * 15.  Below the GPU is used to generat the locations of mutations	*/
			/* in the genome.																									*/
			int *mut_1_lst;
			int *mut_2_lst;
		
			int min = 1;
			int max = 15 * num_reg;			/* Max 30, or get negative numbers with bit arithematic	*/
		
			/* calc_mut_cl sets up and calls the GPU kernel that generates the sites that mutate.							*/
			calc_mut_cl(block_tot_k_1, &mut_1_lst, min, max, rand_uniform_1,
								context_mut, command_queue_mut, program_mut, kernel_mut,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			calc_mut_cl(block_tot_k_2, &mut_2_lst, min, max, rand_uniform_2,
								context_mut, command_queue_mut, program_mut, kernel_mut,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			delete[] rand_uniform_1;
			
			/* Here the GPU is used to generated uniform deviates that will be used to determine which lineage wins			*/
			/* selection events.																							*/
			calc_rand_uniform_cl(block_size, &rand_uniform_1, 
								context_rnd_uniform, command_queue_rnd_uniform, program_rnd_uniform, kernel_rnd_uniform,
								gpu_local_item_size, gpu_global_item_size, gpu_num_cores);
			
			/* Here the section of the graph that will be sent to the GPU to calculate phenotypes and genotypes is generated.	*/
			int *block_ancestor1 = new int[block_size];
			int *block_ancestor2 = new int[block_size];
			int *block_intr_ind = new int[block_size * num_intr];
		
			float *block_x = new float[block_size * num_dim];
			int *block_x_updated = new int[block_size];
			
			int *block_genotype_1 = new int[block_size * num_reg];
			int *block_genotype_2 = new int[block_size * num_reg];
			
			for(int i = 0; i <= block_size - 1; i++)	{
				
				if(ancestor1[i + block_start] == -1)
					block_ancestor1[i] = -1;
				else
					block_ancestor1[i] = ancestor1[i + block_start] - block_start;
				
				if(ancestor2[i + block_start] == -1)
					block_ancestor2[i] = -1;
				else
					block_ancestor2[i] = ancestor2[i + block_start] - block_start;
			
				for(int ind = 0; ind <= num_intr - 1; ind++)	{

					if(intr_ind[i * num_intr + ind + block_start * num_intr] == -1)
						block_intr_ind[i * num_intr + ind] = -1;
					else
						block_intr_ind[i * num_intr + ind] = 
							intr_ind[i * num_intr + ind + block_start * num_intr] - block_start;

				}

				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					block_x[i * num_dim + dim] = x[i * num_dim + dim + block_start * num_dim];

				}

				block_x_updated[i] = x_updated[i + block_start];
				
				for(int j = 0; j<= num_reg - 1; j++)	{
					block_genotype_1[i * num_reg + j] = genotype_1[(i + block_start) * num_reg + j];
					block_genotype_2[i * num_reg + j] = genotype_2[(i + block_start) * num_reg + j];
				}
			}
			
			/* Here calc_x_cl sets up and calls the GPU kernel that works up through a graph section and calculates	*/
			/* marker genotypes, phenotypes and which lineages when selection events								*/
			calc_x_cl(block_size, block_start, block_stop,
						block_x, block_x_updated, 
						block_ancestor1, block_ancestor2, block_intr_ind,
						num_dim, num_intr,
						delta_x_1, delta_x_2,
						k_1, k_2, mut_1_lst, mut_2_lst, 
						block_genotype_1, block_genotype_2, num_reg,
						block_tot_k_1, block_tot_k_2, block_sum_k_1, block_sum_k_2,
						context_calc_x,command_queue_calc_x,program_calc_x,kernel_calc_x,
						gpu_local_item_size, gpu_global_item_size, gpu_num_cores,graph_cycles_mult,
						a_1, a_2, a_3,
						rand_uniform_1, alpha, K);
			
			/* Block information is copied back to the base data objects that represent the graph.					*/			
			for(int i = 0; i <= block_size - 1; i++)	{
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[i * num_dim + dim + block_start * num_dim] = block_x[i * num_dim + dim];
				}
				x_updated[i + block_start] = block_x_updated[i];
				
				for(int j = 0; j<= num_reg - 1; j++)	{
					genotype_1[(i + block_start) * num_reg + j] = block_genotype_1[i * num_reg + j];
					genotype_2[(i + block_start) * num_reg + j] = block_genotype_2[i * num_reg + j];
				}
			}
			
			delete[] block_genotype_1;
			delete[] block_genotype_2;
			delete[] block_intr_ind;
			delete[] block_ancestor1;
			delete[] block_ancestor2;
			delete[] block_x_updated;
			delete[] block_x;
			
			delete[] mut_1_lst;
			delete[] mut_2_lst;
			
			delete[] block_sum_k_1;
			delete[] block_sum_k_2;
			
			delete[] k_1;
			delete[] k_2;
			
			delete[] delta_x_1;
			delete[] delta_x_2;
			
			delete[] time_to_anc_1;
			delete[] time_to_anc_2;
			
			delete[] rand_uniform_1;
			delete[] rand_uniform_2;
			
			delete[] block_ancestor;
			
			delete[] prelim_graph_node_time;
			
		}	
		
		delete[] prelim_graph_node_event;
		delete[] x_updated;
		
		int graph_size = prelim_graph->size();
		for(int i = 0; i <= prelim_graph->size() - 1; i++)
			delete prelim_graph->at(i);
		
		for(int i = 0; i <= block_start_stop->size() - 1; i++)
			delete block_start_stop->at(i);
			
		delete prelim_graph;
		delete block_start_stop;

	return graph_size;
}

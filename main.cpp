#include<new>
#include<math.h>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<stdlib.h>
#include<vector>
#include <time.h> 

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include"header.h"
#include"Mersenne.h"
#include "dc.h"
#include "RanDev.h"

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char *argv[])  {
	long seed = atol(argv[1]);
	init_genrand(seed);
	srand(seed);
	
	/* Read in parameters from parameters.dat file.  Note order of parameters in code. */
	std::ifstream infilep("parameters.dat");
		if(!infilep) std::cerr << "Cannot open input file." << std::endl;
	
	int number_reps; 				infilep >> number_reps;
	int sample_size; 				infilep >> sample_size;
	float sigma;					infilep >> sigma;
	int num_intr;					infilep >> num_intr;
	float t_max;					infilep >> t_max;
	
	float x_mu;						infilep >> x_mu;		x_mu = x_mu/2;
	int num_dim;					infilep >> num_dim;
	float g_mu;						infilep >> g_mu;		g_mu = g_mu/2;
	int num_reg;					infilep >> num_reg;
	
	int gpu_local_item_size;		infilep >> gpu_local_item_size;
	int gpu_global_item_size;		infilep >> gpu_global_item_size;
	int gpu_num_cores;				infilep >> gpu_num_cores;
	int graph_cycles_mult;			infilep >> graph_cycles_mult;
	int block_size;					infilep >> block_size;					
    
    float a_1;						infilep >> a_1;
    float a_2;						infilep >> a_2;
	float a_3;						infilep >> a_3;
	float alpha;					infilep >> alpha;
	float K;						infilep >> K;
	
	/* Create GPU kernels		*/
	cl_context context_calc_x;
  	cl_command_queue command_queue_calc_x;
  	cl_program program_calc_x;
  	cl_kernel kernel_calc_x;
  	
	create_kernel_cl(&context_calc_x, &command_queue_calc_x, &program_calc_x, &kernel_calc_x, 
							CL_DEVICE_TYPE_GPU, "cl_files/calc_x_v2.cl", "calc_x");
	
	cl_context context_rnd_uniform;
  	cl_command_queue command_queue_rnd_uniform;
  	cl_program program_rnd_uniform;
  	cl_kernel kernel_rnd_uniform;
	
	
	create_kernel_cl(&context_rnd_uniform, &command_queue_rnd_uniform, &program_rnd_uniform, &kernel_rnd_uniform, 
							CL_DEVICE_TYPE_GPU, "cl_files/mwc64x.cl", "rnd_uniform");
						
	cl_context context_mut;
  	cl_command_queue command_queue_mut;
  	cl_program program_mut;
  	cl_kernel kernel_mut;
	
	create_kernel_cl(&context_mut, &command_queue_mut, &program_mut, &kernel_mut, 
							CL_DEVICE_TYPE_GPU, "cl_files/calc_mut.cl", "calc_mut");
					
	cl_context context_calc_k;
  	cl_command_queue command_queue_calc_k;
  	cl_program program_calc_k;
  	cl_kernel kernel_calc_k;
							
	create_kernel_cl(&context_calc_k, &command_queue_calc_k, &program_calc_k, &kernel_calc_k, 
							CL_DEVICE_TYPE_GPU, "cl_files/calc_k_v2.cl", "calc_k");
						
	cl_context context_time_to_anc;
  	cl_command_queue command_queue_time_to_anc;
  	cl_program program_time_to_anc;
  	cl_kernel kernel_time_to_anc;
							
	create_kernel_cl(&context_time_to_anc, &command_queue_time_to_anc, &program_time_to_anc, &kernel_time_to_anc, 
							CL_DEVICE_TYPE_GPU, "cl_files/time_to_anc.cl", "time_to_anc");
						
	cl_context context_calc_delta_x;
  	cl_command_queue command_queue_calc_delta_x;
  	cl_program program_calc_delta_x;
  	cl_kernel kernel_calc_delta_x;
							
	create_kernel_cl(&context_calc_delta_x, &command_queue_calc_delta_x, &program_calc_delta_x, &kernel_calc_delta_x, 
							CL_DEVICE_TYPE_GPU, "cl_files/calc_delta_x.cl", "calc_delta_x");
	
	/* Begin replicate simulations of ancestral process */
	for(int rep = 0; rep <= number_reps - 1; rep++)		{																	
		std::cout << "Rep: " << rep << std::endl;
		
		/* Set up pointers to data objects.  These objects are taken as arguments in the gpu_graph() function. */
		int *desc1;
		int *desc2;
		int *ancestor1;
		int *ancestor2;
		int *intr_ind;
		
		float *x;
		int *genotype_1;
		int *genotype_2;
		
		int graph_size;
		
		/* Generate ancestral graph for a replicate */
		graph_size = gpu_graph(sample_size, sigma, num_intr, t_max, x_mu, num_dim, g_mu, num_reg,
				gpu_local_item_size, gpu_global_item_size, gpu_num_cores,
				graph_cycles_mult, block_size,
				a_1, a_2, a_3, alpha, K,
				desc1, desc2, ancestor1, ancestor2, intr_ind,
				x, genotype_1, genotype_2,
				
				context_calc_x, command_queue_calc_x, 
				program_calc_x, kernel_calc_x,
				
				context_rnd_uniform, command_queue_rnd_uniform,
  				program_rnd_uniform, kernel_rnd_uniform,
  				
  				context_mut, command_queue_mut,
  				program_mut, kernel_mut,
  				
  				context_calc_k, command_queue_calc_k,
  				program_calc_k, kernel_calc_k,
  				
  				context_time_to_anc, command_queue_time_to_anc,
  				program_time_to_anc, kernel_time_to_anc,
  				
  				context_calc_delta_x, command_queue_calc_delta_x,
  				program_calc_delta_x, kernel_calc_delta_x
				
				);
        
        std::cout << "Exited gpu_graph" << std::endl;
        
        /* Output to file information from graph, including phenotypes for every 100 node in graph, as well as */
        /* the sample phenotypes and genotypes 																	*/
        FILE *file_ptr;
		char filename[100];
		sprintf(filename,"Data.out");
		file_ptr = fopen(filename,"a");
		if(file_ptr == NULL)	{
			printf("Cannot open file.\n");
			exit(8);
		}
		fprintf(file_ptr,"[");
		for(int i = 0; i <= graph_size - 2; i++)	{
			if(i % 100 == 0) {
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					fprintf(file_ptr, "%f,",x[i * num_dim + dim]);
				}
			}
		}
		for(int dim = 0; dim <= num_dim - 1; dim++)	{
			if(dim < num_dim -1)
        		fprintf(file_ptr, "%f,",x[(graph_size - 1) * num_dim + dim]);
        	else
        		fprintf(file_ptr, "%f",x[(graph_size - 1) * num_dim + dim]);
        }
        fprintf(file_ptr, "]\n");
		fclose(file_ptr);
		
		sprintf(filename,"Geno.out");
		file_ptr = fopen(filename,"a");
		if(file_ptr == NULL)	{
			printf("Cannot open file.\n");
			exit(8);
		}
		fprintf(file_ptr,"[");
		for(int i = 0; i <= sample_size * num_reg - 2; i++)	{
			fprintf(file_ptr, "%d,",genotype_1[i]);
		}
        fprintf(file_ptr, "%d]\n",genotype_1[sample_size * num_reg - 1]);
		fclose(file_ptr);
		
		sprintf(filename,"Pheno.out");
		file_ptr = fopen(filename,"a");
		if(file_ptr == NULL)	{
			printf("Cannot open file.\n");
			exit(8);
		}
		fprintf(file_ptr,"[");
		for(int i = 0; i <= sample_size - 2; i++)	{
			for(int dim = 0; dim <= num_dim - 1; dim++)	{
				fprintf(file_ptr, "%f,",x[i * num_dim + dim]);
			}
		}
		for(int dim = 0; dim <= num_dim - 1; dim++)	{
        	if(dim < num_dim -1)
        		fprintf(file_ptr, "%f,",x[(sample_size - 1) * num_dim + dim]);
        	else
        		fprintf(file_ptr, "%f",x[(sample_size - 1) * num_dim + dim]);
        }
        fprintf(file_ptr, "]\n");
		fclose(file_ptr);
		
		delete[] desc1;
		delete[] desc2;
		delete[] ancestor1;
		delete[] ancestor2;
		delete[] intr_ind;
		
		delete[] x;
		delete[] genotype_1;
		delete[] genotype_2; 
        
	}
	
return 0;
}	

	

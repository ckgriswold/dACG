#include<vector>
#include "dc.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct Node {
	double time;
	
	struct Node *desc1;
	struct Node *desc2;
	struct Node *ancestor1;
	struct Node *ancestor2;
	
	struct Node *intr_ind;
	
	float delta_x_1;
	float delta_x_2;
	float x;
	
	int delta_x_1_updated;
	int delta_x_2_updated;
	int x_updated;
	
	float x_from_anc_1;
	float x_from_anc_2;
	float x_from_anc_1_updated;
	float x_from_anc_2_updated;
	
	int sp;
};

struct Available_Node  {
	struct Node *address;
	struct Available_Node *next_node;
	struct Available_Node *previous_node;
};

void make_prelim_graph(std::vector< float* > *tree, int sample_size, float sigma, int num_intr, float t_max, int *num_coal, int *num_sel, int *num_other,
						int block_size, std::vector< int* > *block_start_stop);

void connect_graph(float *prelim_graph_events, int graph_size, int num_dim, int num_intr, int *desc1, int *desc2, int *ancestor1, int *ancestor2,
						int *intr_ind, int *x_updated, float *x, float *rand_num);

void calc_x_cl(int block_size, int block_start, int block_stop, float *x, int *x_updated, int *ancestor1, int *ancestor2, int *intr_ind,
				int num_dim, int num_intr, float *delta_x_1, float *delta_x_2,
				int *k_1, int *k_2, int *mut_1, int *mut_2, int *genotype_1, int *genotype_2, int num_reg,
				int num_mut_1, int num_mut_2, int *sum_k_1, int *sum_k_2,
				cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
				int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores,
                int graph_cycles_mult,
                float a_1, float a_2, float a_3,
                float *rand_uniform_1, float alpha, float K);

void create_kernel_cl(cl_context *context, cl_command_queue *command_queue, cl_program *program, cl_kernel *kernel, cl_device_type CL_DEVICE_TYPE_X,
		char *kernel_file, char *kernel_func);	
					
void calc_rand_uniform_cl(int graph_size, float **delta_rand_uniform,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores);

void time_to_anc_cl(float *prelim_graph_node_time, int block_size, int block_start, int *ancestor, float **time_to_anc,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores);
								
void calc_k_cl(float *time_to_anc, int graph_size, float *mu, float *rnd_uniform_1, float *rnd_uniform_2, int **k,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores);

void calc_delta_x_cl(int *k, int num_dim, float *rnd_uniform_1, float *rnd_uniform_2, int graph_size, float **delta_x,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores);
								
void calc_mut_cl(int num_k, int **mut_lst, int min, int max, float *rnd_uniform,
							cl_context context, cl_command_queue command_queue, cl_program program, cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores);

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
				
				);
/* Sets up a call to the calc_delta_x.cl GPU kernel */

#include <stdlib.h>
#include <stdio.h>
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

void calc_delta_x_cl(int *k, int num_dim, float *rnd_uniform_1, float *rnd_uniform_2, int graph_size, float **delta_x,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores) {
std::cout << graph_size << std::endl;
  cl_int ret;
		
  /* Shared parameters across parallel processes 	*/
  size_t global_item_size, local_item_size;
  
  local_item_size = gpu_local_item_size;
  int k_item_size = (graph_size - graph_size % local_item_size + local_item_size);
  global_item_size = (graph_size * num_dim - (graph_size * num_dim) % local_item_size + local_item_size);
  	
  /* Generate input and output buffers and set kernel arguments */		
  cl_mem k_buffer;
  cl_mem output_buffer;
  cl_mem rnd_uniform_1_buffer;
  cl_mem rnd_uniform_2_buffer;
  
  /* Create input buffer */
  k_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*k_item_size, k, &ret);			//std::cout << ret << std::endl;
  rnd_uniform_1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*global_item_size, rnd_uniform_1, &ret);
  rnd_uniform_2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*global_item_size, rnd_uniform_2, &ret);
  
  /* Create output buffer */
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*global_item_size, NULL, &ret);	//std::cout << ret << std::endl;
  float *result = new float[global_item_size];
  
  /* Set Kernel Arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&k_buffer); 										//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&rnd_uniform_1_buffer); 							//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&rnd_uniform_2_buffer); 							//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&output_buffer); 									//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 4, sizeof(int), &graph_size);
  ret = clSetKernelArg(kernel, 5, sizeof(int), &num_dim);

  /* Get results	*/	
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);			//std::cout << ret << std::endl;  
  ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, sizeof(float)*global_item_size, result, 0, NULL, NULL);	//std::cout << ret << std::endl;

  *delta_x = result;
  
  ret = clReleaseMemObject(output_buffer);
  ret = clReleaseMemObject(k_buffer);
  ret = clReleaseMemObject(rnd_uniform_1_buffer);
  ret = clReleaseMemObject(rnd_uniform_2_buffer);
  
  return;
}
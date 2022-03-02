/* Sets up a call to the time_to_anc.cl GPU kernel */


#include <stdlib.h>
#include <stdio.h>
#include<new>
#include<vector>
#include<time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include"header.h"

void time_to_anc_cl(float *prelim_graph_node_time, int block_size, int block_start, int *ancestor, float **time_to_anc,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores) {

  cl_int ret;
		
  /* Shared parameters across parallel processes 	*/
  size_t global_item_size, local_item_size;
  
  local_item_size = gpu_local_item_size;
  global_item_size = (block_size - block_size % local_item_size + local_item_size);
  
  /* Generate input and output buffers and set kernel arguments */		
  cl_mem time_buffer;
  cl_mem ancestor_buffer;
  cl_mem output_buffer;
  
  /* Create input buffer */
  time_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*block_size, prelim_graph_node_time, &ret);		//std::cout << ret << std::endl;
  ancestor_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size, ancestor, &ret);					//std::cout << ret << std::endl;
  
  /* Create output buffer */
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*global_item_size, NULL, &ret);	//std::cout << ret << std::endl;
  float *result = new float[global_item_size];
  
  /* Set Kernel Arguments */
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&time_buffer); 									//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 1, sizeof(cl_int), &block_size); 											//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&ancestor_buffer); 								//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&output_buffer); 									//std::cout << ret << std::endl;

  /* Get results	*/	
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);				//std::cout << ret << std::endl;  
  ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, sizeof(float)*global_item_size, result, 0, NULL, NULL);	//std::cout << ret << std::endl;

  *time_to_anc = result;
  
  ret = clReleaseMemObject(output_buffer);
  ret = clReleaseMemObject(time_buffer);
  ret = clReleaseMemObject(ancestor_buffer);

  return;
}
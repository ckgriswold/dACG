/* Sets up a call to the calc_mut.cl GPU kernel */

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

void calc_mut_cl(int num_k, int **mut_lst, int min, int max, float *rnd_uniform,
							cl_context context, cl_command_queue command_queue, cl_program program, cl_kernel kernel,
								int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores) {

  cl_int ret;
  		
  /* Shared parameters across parallel processes 	*/
  size_t global_item_size, local_item_size;
  local_item_size = 1;
  global_item_size = gpu_num_cores;
  
  int tot_num_rand = (num_k - num_k%global_item_size + global_item_size);							//std::cout << tot_num_rand << std::endl;
  int num_rand = tot_num_rand/global_item_size;																//std::cout << num_rand << std::endl;
  
  /* Generate input and/or output buffers and set kernel arguments */		
  cl_mem rnd_uniform_buffer;
  cl_mem output_buffer;
  
  rnd_uniform_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*tot_num_rand, rnd_uniform, &ret);		//std::cout << ret << std::endl;
  
  /* Create output buffer */
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int)*tot_num_rand, NULL, &ret);	//std::cout << ret << std::endl;
  int *result = new int[tot_num_rand];
  
  /* Set Kernel Arguments */
  unsigned long seed = rand();																				//std::cout << seed << std::endl;
  
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), &num_rand); 												//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &seed); 												//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buffer); 									//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 3, sizeof(cl_int), &min); 									//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 4, sizeof(cl_int), &max); 									//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&rnd_uniform_buffer); 								//std::cout << ret << std::endl;

  /* Get results	*/	
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);			//std::cout << ret << std::endl;  
  ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, sizeof(int)*tot_num_rand, result, 0, NULL, NULL);	//std::cout << ret << std::endl;

  *mut_lst = result;
  
  ret = clReleaseMemObject(output_buffer);
  ret = clReleaseMemObject(rnd_uniform_buffer);
  
  return;
}
/* Sets up a call to the mwc64x.cl GPU kernel 							*/
/* Uses the mwc64x generator of David Thomas							*/
/* http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html		*/


#include <stdlib.h>
#include <stdio.h>
#include<new>
#include<iostream>
#include <time.h> 

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "header.h"
#include "dc.h"

typedef struct mt_struct_s {
  uint aaa;
  int mm,nn,rr,ww;
  uint wmask,umask,lmask;
  int shift0, shift1, shiftB, shiftC;
  uint maskB, maskC;
} mt_struct_cl;

void calc_rand_uniform_cl(int graph_size, float **delta_rand_uniform,
							cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
							int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores) {

  cl_int ret;
  		
  /* Shared parameters across parallel processes 	*/
  size_t global_item_size, local_item_size;
  local_item_size = 1;
  global_item_size = gpu_num_cores;
  
  int tot_num_rand = (graph_size - graph_size%global_item_size + global_item_size);							//std::cout << tot_num_rand << std::endl;
  int num_rand = tot_num_rand/global_item_size;																//std::cout << num_rand << std::endl;
  
  /* Generate input and/or output buffers and set kernel arguments */		
  cl_mem output_buffer;
  
  /* Create output buffer */
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*tot_num_rand, NULL, &ret);	//std::cout << ret << std::endl;
  float *result = new float[tot_num_rand];
 
  /* Set Kernel Arguments */
  unsigned long seed = rand();																				//std::cout << seed << std::endl;
  
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), &num_rand); 												//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &seed); 												//std::cout << ret << std::endl;
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&output_buffer); 									//std::cout << ret << std::endl;

  /* Get results	*/	
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);			//std::cout << ret << std::endl;  
  ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, sizeof(float)*tot_num_rand, result, 0, NULL, NULL);	//std::cout << ret << std::endl;

  *delta_rand_uniform = result;
  
  ret = clReleaseMemObject(output_buffer);

  return;
}
/* Sets up a call to the calc_x_v2.cl GPU kernel */

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

void calc_x_cl(int block_size, int block_start, int block_stop,
				float *x, int *x_updated, 
				int *ancestor1, int *ancestor2, int *intr_ind,
				int num_dim, int num_intr,
				float *delta_x_1, float *delta_x_2,
				int *k_1, int *k_2, int *mut_1, int *mut_2, int *genotype_1, int *genotype_2, int num_reg,
				int num_mut_1, int num_mut_2, int *sum_k_1, int *sum_k_2,
				cl_context context,cl_command_queue command_queue,cl_program program,cl_kernel kernel,
				int gpu_local_item_size, int gpu_global_item_size, int gpu_num_cores,
                int graph_cycles_mult,
                float a_1, float a_2, float a_3,
                float *rand_uniform, float alpha, float K) {

printf("Calculating phenotypes and genotypes\n");

  cl_int ret;
		
  /* Shared parameters across parallel processes 	*/
  size_t global_item_size, local_item_size, k_buffer_size, num_mut_1_buffer_size, num_mut_2_buffer_size, delta_x_buffer_size, rand_buffer_size;
  global_item_size = gpu_global_item_size; /* 3584 on Tesla P100, 12288 on Intel Iris 6100	*/
  local_item_size = gpu_local_item_size;  /* 64 on Tesla P100, 256 on Intel Iris 6100		*/
  
  k_buffer_size = (block_size - block_size % local_item_size + local_item_size);
  num_mut_1_buffer_size = (num_mut_1 - num_mut_1 % gpu_num_cores + gpu_num_cores);
  num_mut_2_buffer_size = (num_mut_2 - num_mut_2 % gpu_num_cores + gpu_num_cores);
  delta_x_buffer_size = (block_size * num_dim - (block_size * num_dim) % local_item_size + local_item_size);
  rand_buffer_size = (block_size - block_size % gpu_num_cores + global_item_size);
  		
	int *gid_to_node_translator;
	gid_to_node_translator = new int[global_item_size];
	
	for(int i = 0; i <= global_item_size - 1; i++)	gid_to_node_translator[i] = block_size - i - 1;
	
		/* Generate input buffers and set kernel arguments. */		
  		cl_mem x_buffer;
  		cl_mem delta_x_1_buffer;
  		cl_mem delta_x_2_buffer;
  		cl_mem ancestor1_buffer;
  		cl_mem ancestor2_buffer;
  		cl_mem intr_ind_buffer;
  		cl_mem x_updated_buffer;
  		cl_mem k_1_buffer;
  		cl_mem k_2_buffer;
  		cl_mem mut_1_buffer;
  		cl_mem mut_2_buffer;
  		cl_mem genotype_1_buffer;
  		cl_mem genotype_2_buffer;
  		cl_mem gid_to_node_translator_buffer;
  		cl_mem sum_k_1_buffer;
  		cl_mem sum_k_2_buffer;
  		cl_mem rand_uniform_buffer;
  		

  		x_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*block_size*num_dim, x, &ret); 						if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		delta_x_1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*delta_x_buffer_size, delta_x_1, &ret);  		if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		delta_x_2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*delta_x_buffer_size, delta_x_2, &ret); 		if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		ancestor1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size, ancestor1, &ret); 			if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		ancestor2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size, ancestor2, &ret); 			if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		intr_ind_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size*num_intr, intr_ind, &ret); 			if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		x_updated_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size, x_updated, &ret); 			if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		k_1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*k_buffer_size, k_1, &ret);						if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		k_2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*k_buffer_size, k_2, &ret); 						if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		mut_1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*num_mut_1_buffer_size, mut_1, &ret);			if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		mut_2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*num_mut_2_buffer_size, mut_2, &ret); 			if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		genotype_1_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size*num_reg, genotype_1, &ret);		if(ret != CL_SUCCESS) printf("Buffer problem\n");
  		genotype_2_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size*num_reg, genotype_2, &ret); 		if(ret != CL_SUCCESS) printf("Buffer problem\n");
		gid_to_node_translator_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*global_item_size, 
  															gid_to_node_translator, &ret); 															if(ret != CL_SUCCESS) printf("Buffer problem\n");
		
		sum_k_1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size, sum_k_1, &ret);					if(ret != CL_SUCCESS) printf("Buffer problem\n");
		sum_k_2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*block_size, sum_k_2, &ret);					if(ret != CL_SUCCESS) printf("Buffer problem\n");
		
		rand_uniform_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*rand_buffer_size, rand_uniform, &ret);		if(ret != CL_SUCCESS) printf("Buffer problem\n");
		
printf("Size of buffer to GPU: %lu\n",sizeof(cl_float)*block_size*num_dim + 2 * sizeof(cl_float)*delta_x_buffer_size + 
										 2 * sizeof(cl_int)*block_size + sizeof(cl_int)*block_size*num_intr +
										 sizeof(cl_int)*block_size + 2* sizeof(cl_int)*k_buffer_size +
										 sizeof(cl_int)*num_mut_1_buffer_size + sizeof(cl_int)*num_mut_2_buffer_size +
										 2 * sizeof(cl_int)*block_size*num_reg + sizeof(cl_int)*global_item_size +
										 2 * sizeof(cl_int)*block_size + sizeof(cl_float)*rand_buffer_size);
printf("Block size: %d\n",block_size); 
		 		
  		/* Set Kernel Arguments for all buffers */
  		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_buffer); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta_x_1_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &delta_x_2_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &ancestor1_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &ancestor2_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &intr_ind_buffer); 					if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &x_updated_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), &gid_to_node_translator_buffer);	if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 8, sizeof(cl_int), &global_item_size); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		
  		ret = clSetKernelArg(kernel, 9, sizeof(cl_mem), &k_1_buffer); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 10, sizeof(cl_mem), &k_2_buffer); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 11, sizeof(cl_mem), &mut_1_buffer); 					if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 12, sizeof(cl_mem), &mut_2_buffer); 					if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 13, sizeof(cl_mem), &genotype_1_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 14, sizeof(cl_mem), &genotype_2_buffer); 				if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 15, sizeof(cl_float), &a_1); 							if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 16, sizeof(cl_float), &a_2); 							if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 17, sizeof(cl_float), &a_3); 							if(ret != CL_SUCCESS) printf("Arg problem\n");
  		
  		ret = clSetKernelArg(kernel, 18, sizeof(cl_mem), &sum_k_1_buffer); 					if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 19, sizeof(cl_mem), &sum_k_2_buffer); 					if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 20, sizeof(cl_int), &num_reg); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
  		ret = clSetKernelArg(kernel, 21, sizeof(cl_int), &num_dim); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
		ret = clSetKernelArg(kernel, 22, sizeof(cl_int), &num_intr); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
		ret = clSetKernelArg(kernel, 23, sizeof(cl_mem), &rand_uniform_buffer); 			if(ret != CL_SUCCESS) printf("Arg problem\n");
		ret = clSetKernelArg(kernel, 24, sizeof(cl_float), &alpha); 						if(ret != CL_SUCCESS) printf("Arg problem\n");
		ret = clSetKernelArg(kernel, 25, sizeof(cl_float), &K); 							if(ret != CL_SUCCESS) printf("Arg problem\n");

	int count = 0;
	int check = 0;	
    int check_value = (int) block_size/global_item_size * graph_cycles_mult;
	int num_check = 0;
	
	while(1)	{
		check++;
					
  		/* Get results	*/
  		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  		
  		/* Check to see if graph has been calculated, but only after check_value loops	*/
		if(check > check_value)	{
			num_check++;

			ret = clEnqueueReadBuffer(command_queue, gid_to_node_translator_buffer, CL_TRUE, 0, sizeof(int)*global_item_size, 
											gid_to_node_translator, 0, NULL, NULL);

			count = 0;
			for(int i = 0; i <= global_item_size - 1; i++) if(gid_to_node_translator[i] >= 0)	count++;
            
            printf("Calc x stats, Num positive nodes in cycle: %d, cycle: %d \n",count,check);
            /*for(int i = 0; i <= global_item_size - 1; i++) if(gid_to_node_translator[i] <= 0) printf("%d ",gid_to_node_translator[i]);
			printf("\n");*/
            
            if(count < 1)
                break;
		}
	}

	//printf("%d\n",num_check);
		
		/* Read back x */
		ret = clEnqueueReadBuffer(command_queue, x_buffer, CL_TRUE, 0, sizeof(float)*block_size*num_dim, x, 0, NULL, NULL);						//printf("%d\n",ret);	
		ret = clEnqueueReadBuffer(command_queue, x_updated_buffer, CL_TRUE, 0, sizeof(int)*block_size, x_updated, 0, NULL, NULL);
		ret = clEnqueueReadBuffer(command_queue, genotype_1_buffer, CL_TRUE, 0, sizeof(int)*block_size*num_reg, genotype_1, 0, NULL, NULL);		//printf("%d\n",ret);
		ret = clEnqueueReadBuffer(command_queue, genotype_2_buffer, CL_TRUE, 0, sizeof(int)*block_size*num_reg, genotype_2, 0, NULL, NULL);		//printf("%d\n",ret);		
		
  		/* Release Buffers */
		ret = clReleaseMemObject(x_buffer);
		ret = clReleaseMemObject(delta_x_1_buffer);
		ret = clReleaseMemObject(delta_x_2_buffer);
		ret = clReleaseMemObject(ancestor1_buffer);
		ret = clReleaseMemObject(ancestor2_buffer);
		ret = clReleaseMemObject(intr_ind_buffer);
		ret = clReleaseMemObject(x_updated_buffer);
		ret = clReleaseMemObject(gid_to_node_translator_buffer);
		ret = clReleaseMemObject(k_1_buffer);
		ret = clReleaseMemObject(k_2_buffer);
        ret = clReleaseMemObject(mut_1_buffer);
		ret = clReleaseMemObject(mut_2_buffer);
		ret = clReleaseMemObject(genotype_1_buffer);
		ret = clReleaseMemObject(genotype_2_buffer);
		
		ret = clReleaseMemObject(sum_k_1_buffer);
		ret = clReleaseMemObject(sum_k_2_buffer);
		
		ret = clReleaseMemObject(rand_uniform_buffer);
		
	delete gid_to_node_translator;
	
	return;
}

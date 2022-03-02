/* A function that generates GPU kernels		*/

#include <stdlib.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif	
 
#define MAX_SOURCE_SIZE (0x100000)
 
void create_kernel_cl(cl_context *context, cl_command_queue *command_queue, cl_program *program, cl_kernel *kernel, cl_device_type CL_DEVICE_TYPE_X,
		char *kernel_file, char *kernel_func)
{
  
  cl_int ret;
  
  /* 	Get Platform IDs   */
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_platforms;
  
  clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  
  /* 	Get Device IDs		*/
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_X, 1, &device_id,&ret_num_devices);
  	if(ret != CL_SUCCESS) printf("Problem getting Device IDs\n");
  
  /* 	Create Context		*/
  *context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  	if(ret != CL_SUCCESS) printf("Problem creating Context\n");
  
  /*	Create Command Queue	*/
  *command_queue = clCreateCommandQueue(*context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
  	if(ret != CL_SUCCESS) printf("Problem creating Command Queue\n");
  	
  /* Build Program*/
  char *kernel_src_str;
  size_t kernel_code_size;
  
  FILE *fp;
  fp = fopen(kernel_file, "r");
  kernel_src_str = (char*)malloc(MAX_SOURCE_SIZE);
  kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
  
  *program = clCreateProgramWithSource(*context, 1, (const char **)&kernel_src_str, (const size_t *)&kernel_code_size, &ret);
  ret = clBuildProgram(*program, 1, &device_id, "", NULL, NULL);
   
  if(ret != CL_SUCCESS)
    {
        printf("clBuildProgram has failed\n");
        
        size_t len = 0;
        ret = clGetProgramBuildInfo(*program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *buffer;
		buffer = (char *) calloc(len, sizeof(char));
		ret = clGetProgramBuildInfo(*program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		
        printf("%s\n", buffer);
        exit(1);
    }
  
  *kernel = clCreateKernel(*program, kernel_func, &ret);
   		if(ret != CL_SUCCESS) printf("problem with creating kernel poidev\n");
 
  return;
  }
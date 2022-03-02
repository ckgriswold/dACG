/*  Box-Muller Gaussian generator 																*/
/* https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution */

__kernel void calc_delta_x(__global int *k, __global float *U, __global float *V, __global float *result, int graph_size, int num_dim)
{

	int gid = get_global_id(0);
	
	int phase;
	
	phase = gid % 2;
	
	int gid_k;
	
	
	
	gid_k = convert_int_rtz( floor( convert_float(gid) / convert_float(num_dim) ) );
	
	if(gid_k > graph_size)	{
		result[gid] = 0.0;
		return;
	}
	
	if(U[gid] <= 0.0 || V[gid] <= 0.0)	{
		result[gid] = 0.0;
		return;
	}

	if(phase == 0) 
		result[gid] = pow((float) -2.0 * log(U[gid]), (float) 0.5) * sin( (float) (2.0 * 3.141592654 * V[gid])) * pow( (float) k[gid_k], (float) 0.5 );
	else
		result[gid] = pow((float) -2.0 * log(U[gid]), (float) 0.5) * cos( (float) (2.0 * 3.141592654 * V[gid])) * pow( (float) k[gid_k], (float) 0.5 );
	
	/*if(phase == 0) 
		result[gid] = (float) gid_k;
	else
		result[gid] = (float) gid_k;
	*/
	
	return;
}

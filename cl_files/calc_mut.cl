/* Returns the site of a mutation  */

__kernel void calc_mut(int n, ulong baseOffset, __global int *acc, int min, int max, __global float *rand_num)
{
	
	int gid = get_global_id(0);

    for(int i=0;i<n;i++){ 	
		if (rand_num[gid * n + i] < 1.0)	{
			float ran_dev;
			ran_dev = rand_num[gid * n + i] * (max + 1 - min);
			acc[gid * n + i] = min + convert_int_rtz(floor(ran_dev));
		}
		else
			acc[gid * n + i] = max;
    	}
	return;
}
/* Returns the time between two nodes 	*/

__kernel void time_to_anc(__global float *prelim_graph_node_time, int block_size, __global int *ancestor, __global float *result) {
	
	int gid = get_global_id(0);
	
	if(ancestor[gid] > block_size - 1)
		result[gid] = 0.0;
	else if(ancestor[gid] >= 0)
		result[gid] = prelim_graph_node_time[ancestor[gid]] - prelim_graph_node_time[gid];
	else
		result[gid] = 0.0;
		
	return;
}
/* Calculates a possion deviate giving the number of mutations along an edge in the graph using the expected number of mutations.	*/
/* It uses the cummulative probability approach and normal approximation.															*/

float poisson( int k, float mean ) { 
  float p = exp(-mean);
  float f = 1;
  for ( int i=0 ; i<k ; i++ ) f *= mean/(i+1);     
  return p*f;
}

__kernel void calc_k(__global float *time, float xm, __global float *U, __global float *V, __global uint *result, int graph_size)
{
 
 int gid = get_global_id(0);
 
 int phase;
	
 phase = gid % 2;

if(gid < graph_size && time[gid] > 0.0)	{
 
 	if(xm * time[gid] < 20)	{

		int i = 0;
		float cdf = 0;
 		float tmp;
		do	{
 			tmp = poisson(i,xm * time[gid]);
 			cdf = cdf + tmp;
 			i = i + 1;
	
			if(cdf >= U[gid]) break;
		 } while(tmp > 0.000001 || i < convert_int(xm * time[gid]) * 4);
 
	 	result[gid] = i - 1;
	 }
	else if(xm * time[gid] < 1000)	{
		float tmp;
	 
	 	if(phase == 0) 
			tmp = 0.5 + xm * time[gid] + 
				pow((float) -2.0 * log(U[gid]), (float) 0.5) * sin( (float) (2.0 * 3.141592654 * V[gid])) * pow( xm * time[gid], (float) 0.5 );
		else
			tmp = 0.5 + xm * time[gid] + 
				pow((float) -2.0 * log(U[gid]), (float) 0.5) * cos( (float) (2.0 * 3.141592654 * V[gid])) * pow( xm * time[gid], (float) 0.5 );
		
		result[gid] = convert_int_rtz( tmp );
	}
	else	{
		float tmp;
		
		if(phase == 0) 
			tmp = xm * time[gid] + 
				pow((float) -2.0 * log(U[gid]), (float) 0.5) * sin( (float) (2.0 * 3.141592654 * V[gid])) * pow( xm * time[gid], (float) 0.5 );
		else
			tmp = xm * time[gid] + 
				pow((float) -2.0 * log(U[gid]), (float) 0.5) * cos( (float) (2.0 * 3.141592654 * V[gid])) * pow( xm * time[gid], (float) 0.5 );
	
		result[gid] = convert_int_rtz( tmp );
	}
}
else
 	result[gid] = 0;
 
 return;

}
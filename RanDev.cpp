#include<math.h>
#include<iostream>

/* Generates a random uniform integer in the interval (min,max) */
int RandomUniformInt(float rand_num, int min, int max)  {
	float ran_dev, ran_remainder;
	
	if (rand_num < 1.0)	{
		ran_dev = rand_num * (float(max+1) - float(min));
		return min + int(floor(ran_dev));
	}
	else
		return max;
}
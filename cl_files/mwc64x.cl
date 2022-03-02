/* Generates uniform deviates between 0 and 1

Part of MWC64X by David Thomas, dt10@imperial.ac.uk  This is provided under BSD, full license is with the main package.
See http://www.doc.ic.ac.uk/~dt10/research
*/

#ifndef dt10_mwc32_cl
#define dt10_mwc32_cl

#include "cl_files/mwc64x/mwc64x_rng.cl"
#include "cl_files/mwc64x/mwc64xvec2_rng.cl"
#include "cl_files/mwc64x/mwc64xvec4_rng.cl"
#include "cl_files/mwc64x/mwc64xvec8_rng.cl"

#endif

__kernel void rnd_uniform(int n, ulong baseOffset, __global float *acc)
{
	
	int gid = get_global_id(0);
	
	mwc64x_state_t rng;
    MWC64X_SeedStreams(&rng, baseOffset	, n);

    for(int i=0;i<n;i++){ 	
    	ulong res = MWC64X_NextUint(&rng);
    	acc[gid * n + i] = convert_float(res)/4294967296.0f;
    }

}
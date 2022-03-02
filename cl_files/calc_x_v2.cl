/* Calculates the phenotype and genotype of a node */

__kernel void calc_x(	__global float *x,
						__global const float *delta_x_1, __global const float *delta_x_2, 
						__global const int *ancestor1, __global const int *ancestor2,
						__global const int *intr_ind,
						__global int *x_updated,
						__global int *gid_to_node_translator,
						int interval,
						__global const int *k_1,
						__global const int *k_2,
						__global const int *mut_1,
						__global const int *mut_2,
						__global int *genotype_1,
						__global int *genotype_2,
						float a_1, float a_2, float a_3,
						__global const int *sum_k_1,
						__global const int *sum_k_2,
						int num_reg, int num_dim, int num_intr,
						__global const float *rand_uniform, float alpha, float K)
{
	
	int gid = get_global_id(0);
	
	int j;
	j = gid_to_node_translator[gid];
	
	int mut_1_indx = 0;
	int mut_2_indx = 0;
	
	mut_1_indx = sum_k_1[j];
	mut_2_indx = sum_k_2[j];
	
	int check;
	
	
	if(j > -1)	{
			
			if(ancestor1[j] == -1 && ancestor2[j] == -1)	{
			
				/* A basal node draw from phi(g,x)			*/
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[j * num_dim + dim] = 0.0;
				}
				x_updated[j] = 1;
				
				gid_to_node_translator[gid] = j - interval;
			}
			else if(x_updated[j] > 0)	{
				gid_to_node_translator[gid] = j - interval;
			}
			else if(ancestor2[j] == -1 && x_updated[ancestor1[j]] > 0 && x_updated[j] < 1)	{
			
				/* A node involved in a coalescence event.  Update in a neutral manner.		*/
				
				/* Update the phenotype.													*/
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[j * num_dim + dim] = x[ancestor1[j] * num_dim + dim] + delta_x_1[j * num_dim + dim];
				}
				
				/* Update the genotype.														*/
				for(int reg = 0; reg <= num_reg - 1; reg++)	{
					genotype_1[j * num_reg + reg] = genotype_1[ancestor1[j] * num_reg + reg];
				}
				
				for (int i = 0; i <= k_1[j] - 1; i++) 	{
					int tmpG;
					
					float tmp = convert_float(mut_1[mut_1_indx]);
					float tmp2 = (tmp - 1.0)/15.0;
					
					int reg = convert_int_rtz(floor(tmp2));
					
					int reg_bp = ((mut_1[mut_1_indx]) - reg * 15);
					
					if(reg_bp > 15) reg_bp = 15;
					
					tmpG = genotype_1[j * num_reg + reg] ^ (1<< reg_bp);
					genotype_1[j * num_reg + reg] = tmpG;

					mut_1_indx = mut_1_indx + 1;
				}
				
				x_updated[j] = 1;
				
				gid_to_node_translator[gid] = j - interval;
			}
			else if(ancestor1[j] > -1 && ancestor2[j] > -1 && x_updated[j] < 1)		{
				if(x_updated[ancestor1[j]] > 0 && x_updated[ancestor2[j]] > 0)	{
					check = 0;
					for(int ind = 0; ind <= num_intr - 1; ind++)	{
						if(intr_ind[ancestor1[j] * num_intr + ind] > -1)
							check = check + 1;
					}
					if(check >= num_intr)	{
						check = 0;
						for(int ind = 0; ind <= num_intr - 1; ind++)	{
							if(x_updated[intr_ind[ancestor1[j] * num_intr + ind]] > 0)
								check = check + 1;
						}
						
						/* A node involved in a selection event and its possible ancestors and interacting nodes have been updated 	*/
						
						if(check >= num_intr)	{
							
							/* Update phenotypes of the two possible ancestors to the node											*/
							float tmp1[10], tmp2[10];	
                            for(int dim = 0; dim <= num_dim - 1; dim++)	{
                            	tmp1[dim] = (x[ancestor1[j] * num_dim + dim] + delta_x_1[j * num_dim + dim]);
                            	tmp2[dim] = (x[ancestor2[j] * num_dim + dim] + delta_x_2[j * num_dim + dim]);
                            }
                            
                            
                            /* Update genotypes of the two possible ancestors to the node											*/
                            for(int reg = 0; reg <= num_reg - 1; reg++)	{
								genotype_1[j * num_reg + reg] = genotype_1[ancestor1[j] * num_reg + reg];
								genotype_2[j * num_reg + reg] = genotype_1[ancestor2[j] * num_reg + reg];
							}
                            
                           	int i;
							for (i = 0; i <= k_1[j] - 1; i++) 	{
								int tmpG;
								
								float tmp = convert_float(mut_1[mut_1_indx]);
								float tmp2 = (tmp - 1.0)/15.0;
					
								int reg = convert_int_rtz(floor(tmp2));
								
								int reg_bp = ((mut_1[mut_1_indx]) - reg * 15);
								
								if(reg_bp > 15) reg_bp = 15;

								tmpG = genotype_1[j * num_reg + reg] ^ (1<< reg_bp);
								genotype_1[j * num_reg + reg] = tmpG;

								mut_1_indx = mut_1_indx + 1;
							}
							
							for (i = 0; i <= k_2[j] - 1; i++) 	{
								int tmpG;
								
								float tmp = convert_float(mut_2[mut_2_indx]);
								float tmp2 = (tmp - 1.0)/15.0;
					
								int reg = convert_int_rtz(floor(tmp2));
								
								int reg_bp = ((mut_2[mut_2_indx]) - reg * 15);
								
								if(reg_bp > 15) reg_bp = 15;
					
								tmpG = genotype_2[j * num_reg + reg] ^ (1<< reg_bp);
								genotype_2[j * num_reg + reg] = tmpG;

								mut_2_indx = mut_2_indx + 1;
							}
							
							/* Calculated the number of non - synonomous mutations						*/
							int nonsyn_k_1 = 0;
							for (i = 1; i <= 15; i++) 	{
								if(genotype_1[j * num_reg] & (1 << i))
									nonsyn_k_1 = nonsyn_k_1 + 1;
							}
							
							int nonsyn_k_2 = 0;
							for (i = 1; i <= 15; i++) 	{
								if(genotype_2[j * num_reg] & (1 << i))
									nonsyn_k_2 = nonsyn_k_2 + 1;
							}
							
							/* Calculate relative fitnesses of two possible ancestors					*/
                        	float tmp3,tmp4;	
                        	float tmp3_hold = convert_float(nonsyn_k_1);
                        	float tmp4_hold = convert_float(nonsyn_k_2);
                        	tmp3 = exp(-a_3 * tmp3_hold);
                        	tmp4 = exp(-a_3 * tmp4_hold);
                        	                        	
                        	
                        	for(int dim = 0; dim <= num_dim - 1; dim++)	{
                        		tmp3_hold = tmp3;
                        		tmp3 = tmp3_hold * (1.0 - 1.0 /
                        						(K*exp(-a_1 * pow(tmp1[dim] - 0,2))));
                        	for(int ind = 0; ind <= num_intr - 1; ind++)	{
                        		float tmp_loop;
                        		tmp_loop = x[intr_ind[ancestor1[j] * num_intr + ind] * num_dim + dim];
                        		tmp3_hold = tmp3;
                        		tmp3 = tmp3_hold * (1.0 - exp(-a_2 * pow((tmp1[dim] - tmp_loop),4)) /
                        						(K*exp(-a_1 * pow(tmp1[dim] - 0,2))));
                        	}}
                        	
                        	
                        	for(int dim = 0; dim <= num_dim - 1; dim++)	{
                        		tmp4_hold = tmp4;
                        		tmp4 = tmp4_hold * (1.0 - 1.0 /
                        						(K*exp(-a_1 * pow(tmp2[dim] - 0,2))));
                        	for(int ind = 0; ind <= num_intr - 1; ind++)	{
                        		float tmp_loop;
                        		tmp_loop = x[intr_ind[ancestor2[j] * num_intr + ind] * num_dim + dim];
                        		tmp4_hold = tmp4;
                            	tmp4 = tmp4_hold * (1.0 - exp(-a_2 * pow((tmp2[dim] - tmp_loop),4)) /
                        						(K*exp(-a_1 * pow(tmp2[dim] - 0,2))));
                            }}
                            
                            float w;
                            float repl_test;
                            
                            /* Calculate relative fitness and probability of displacement					*/
                            w = tmp4/tmp3;
                            repl_test = (pow(w,alpha) - 1.0)/(pow(w,alpha));
                            
                            if(rand_uniform[j] >= repl_test || w <= 1.0)	{
                            	for(int dim = 0; dim <= num_dim - 1; dim++)	{
                                	x[j * num_dim + dim] = tmp1[dim];
                                }
                                
                                for(int reg = 0; reg <= num_reg - 1; reg++)	{
									genotype_2[j * num_reg + reg] = genotype_1[j * num_reg + reg];
								}

                            }
                            else	{
                            	for(int dim = 0; dim <= num_dim - 1; dim++)	{
	                            	x[j * num_dim + dim] = tmp2[dim];
	                            }
                            	
                            	for(int reg = 0; reg <= num_reg - 1; reg++)	{
									genotype_1[j * num_reg + reg] = genotype_2[j * num_reg + reg];
								}
								
                            }

                            x_updated[j] = 1;	
					
							gid_to_node_translator[gid] = j - interval;
						}
						else	{
							/* Do nothing	*/
						}
					}
					else	{
						/* Do nothing	*/
					}
				}
				else	{
					/* Not able to update yet, so keep in list	*/
				}
			}
			else	{
				/* Not able to update yet, so keep in list	*/
			}
		}
		else	{
			/* Not able to update yet, so keep in list	*/
		}
}



#include<new>
#include<math.h>
#include<iostream>
#include<vector>

#include"header.h"
#include"Mersenne.h"
#include"RanDev.h"

void connect_graph(float *prelim_graph_events, int graph_size, int num_dim, int num_intr, int *desc1, int *desc2, int *ancestor1, int *ancestor2,
						int *intr_ind, int *x_updated, float *x, float *rand_num) {
			
	std::vector<int> available_nodes;
	int rand_indx = 0;
	
	std::vector<int> tmp_available_nodes;
	int type_3_indicator = 0;
	
	int i = 0;
	while(i <= graph_size - 1) {
		
		if(type_3_indicator == 1 && prelim_graph_events[i] != 3.0)	{
			type_3_indicator = 0;
			
			for(int j = 0; j <= tmp_available_nodes.size() - 1; j++)
				available_nodes.push_back(tmp_available_nodes.at(j));
			
			tmp_available_nodes.clear();
				
		}
		
		if(prelim_graph_events[i] == 0.0)	{
				/* Initialization */
				
				desc1[i] = -1;
				desc2[i] = -1;
				ancestor1[i] = -1;
				ancestor2[i] = -1;
				
				for(int ind = 0; ind <= num_intr - 1; ind++)	{
					intr_ind[i * num_intr + ind] = -1;
				}
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[i * num_dim + dim] = 0.0;
				}
				x_updated[i] = 0;
				
				available_nodes.push_back(i);
				i = i + 1;
		}
		else if(prelim_graph_events[i] == 3.0)	{
				/* Graph Slice	*/
				
				int indx = RandomUniformInt(rand_num[rand_indx],0,available_nodes.size() - 1);	rand_indx = rand_indx + 1;
				desc1[i] = available_nodes.at(indx);
					ancestor1[desc1[i]] = i;
					available_nodes.erase(available_nodes.begin() + indx);
				
				desc2[i] = -1;
				ancestor1[i] = -1;
				ancestor2[i] = -1;
				
				for(int ind = 0; ind <= num_intr - 1; ind++)	{
					intr_ind[i * num_intr + ind] = -1;
				}
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[i * num_dim + dim] = 0.0;
				}
				x_updated[i] = 0;
				
				tmp_available_nodes.push_back(i);
				i = i + 1;		
				
				type_3_indicator = 1;
		}
		else if(prelim_graph_events[i] == 1.0)	{
				/* Coalescent	*/
				
				/* Choose first descendant and link to ancestor */
				int indx = RandomUniformInt(rand_num[rand_indx],0,available_nodes.size() - 1);	rand_indx = rand_indx + 1;
				desc1[i] = available_nodes.at(indx);
					ancestor1[desc1[i]] = i;
					available_nodes.erase(available_nodes.begin() + indx);
				
				/* Choose second descendant	and link to ancestor */
				indx = RandomUniformInt(rand_num[rand_indx],0,available_nodes.size() - 1);		rand_indx = rand_indx + 1;
				desc2[i] = available_nodes.at(indx);
					ancestor1[desc2[i]] = i;
					available_nodes.erase(available_nodes.begin() + indx);

				ancestor1[i] = -1;
				ancestor2[i] = -1;
				
				for(int ind = 0; ind <= num_intr - 1; ind++)	{
					intr_ind[i * num_intr + ind] = -1;
				}
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[i * num_dim + dim] = 0.0;
				}
				x_updated[i] = 0;
				
				available_nodes.push_back(i);
				i = i + 1;
		}		
		else if(prelim_graph_events[i] == 2.0)	{
				/* Selection 	*/
				
				/* Choose node that descends from competing nodes	*/
				int indx = RandomUniformInt(rand_num[rand_indx],0,available_nodes.size() - 1);	rand_indx = rand_indx + 1;
					ancestor1[available_nodes.at(indx)] = i;
					ancestor2[available_nodes.at(indx)] = i + 1;
				
				/* Link first competing node to its descendant */
				desc1[i] = available_nodes.at(indx);
				desc2[i] = -1;
				
				/* Link second competing node to its descendant */
				desc1[i + 1] = available_nodes.at(indx);
				desc2[i + 1] = -1;
				
				/* Initialize ancestors of competing nodes	*/
				ancestor1[i] = -1;
				ancestor2[i] = -1;
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[i * num_dim + dim] = 0.0;
				}
				x_updated[i] = 0;
				
				ancestor1[i + 1] = -1;
				ancestor2[i + 1] = -1;
				
				for(int dim = 0; dim <= num_dim - 1; dim++)	{
					x[(i + 1) * num_dim + dim] = 0.0;
				}
				x_updated[i + 1] = 0;
				
				/* Link competing nodes to their interacting node	*/
				for(int ind = 0; ind <= num_intr - 1; ind++)	{
					intr_ind[i * num_intr + ind] = i + 2 + ind;
					intr_ind[(i + 1) * num_intr + ind] = i + 2 + ind;
				}
				
				/* Initialize interacting node */
				for(int ind = 0; ind <= num_intr - 1; ind++)	{
					desc1[i + 2 + ind] = -1;
					desc2[i + 2 + ind] = -1;
					ancestor1[i + 2 + ind] = -1;
					ancestor2[i + 2 + ind] = -1;
					
					for(int ind2 = 0; ind2 <= num_intr - 1; ind2++)	{
						intr_ind[(i + 2 + ind) * num_intr + ind2] = -1;
					}
				
					for(int dim = 0; dim <= num_dim - 1; dim++)	{
						x[(i + 2 + ind) * num_dim + dim] = 0.0;
					}
					x_updated[i + 2 + ind] = 0;
				}
				
				available_nodes.erase(available_nodes.begin() + indx);	/* Delete descendant node */
				
				available_nodes.push_back(i);	/* Add first competing node */
				i = i + 1;
				available_nodes.push_back(i);	/* Add second competing node */
				i = i + 1;
				
				for(int ind = 0; ind <= num_intr - 1; ind++)	{
					available_nodes.push_back(i);	/* Add interacting node	*/
					i = i + 1;
				}
		}
		else	{
			std::cout << "Error determining if a coal or selection event occurred in Connect Graph." << std::endl;
			exit(8);
		}
	}
	
	std::cout << "Exiting connect_graph" << std::endl;
	return;
}		
The program can be compiled using the following command:

Apple OS

g++ -o name_of_executable *.c *.cpp -framework OpenCL

Although Apple has deprecated OpenCL, we have had success compiling and running on Monterey using Xcode command-line tools on an Apple M1 chip.  dACG does not use functionality beyond release 1.2 of OpenCL, which Apple appears to support, presently.

Linux & Unix

g++ -o name_of_executable *.c *.cpp -lOpenCL

Note, this assumes the OpenCL libraries are installed and within an accessible PATH in Linux and Unix.

*********************************

The program is executed as

./name_of_executable seed_for_random_number_generation

The seed_for_random_number_generation is a long integer.

Note the cl_files folder must be copied into folders with the executable for the program to run.  

This cl_files folder contains the GPU kernels. 

*********************************

The program expects a parameter.dat file in the same folder as the executable.  The order of parameters in the file is required to be

number of replicates
sample size
sigma divided by 2
number of interacting individuals per potential displacement event
time to colonization event
theta for phenotype
number of dimensions of phenotype
theta for marker genotype
number of 15 bp genomic regions
number of local jobs per core on GPU (current program assumes this is equal to 1)
number of total jobs per GPU (current program assumes this is equal to the number of cores in GPU, see next)
number of cores on GPU
a factor by which the calc_x_cl_v2.cpp checks whether calc_x.cl kernel is finished processing all nodes in section (see note [1] below)
number of nodes in graph to be processed by the GPU (see note [2] below)
a3 (parameter determining the strength of selection on nonsynonous mutations)
a2 (parameter determining the breadth of competition)
a1 (parameter determining the strength of stabilizing selection)
alpha (>= 1 and determines the probability of displacement given a value of relative fitness)
K0 (parameter that determines maximum viability an in the context of stabilizing selection)

[1] Use the example parameter.dat files as a starting point.  If the program returns a lot of the following output

"Calc x stats, Num positive nodes in cycle: 0, cycle: some_number that continually increases . . ."

then this parameter is too small.  Increment up.

[2] The program returns the following information

"Size of buffer to GPU: some_number"

if some_number is greater than the memory capacity of your GPU or even a bit less (in bytes), then decrease this parameter value.  Often the program 
will just exit if the capacity of the GPU is breached, but if you are close to the capacity of the GPU, it may just skip over calculations 
or truncate calculations.  (See below and associated paper for strategies for running simulations).

*****************************

Other notes:

[1] If you get the message

"The combination of graph size and number of genomic regions is too large.  Reduce the rate of selection, number of interacting 
individuals or number of genomic regions."

The graph is too large for your computer architecture and system (non-GPU) memory.  In the future I will update the program such that it
does not keep the entire graph in the system memory.

The memory requirement of the program is approximately equal to the number of nodes in the graph times the sum of the dimensionality of the phenotype and the
number of genomic regions.  The number of nodes in the graph increases with an increase in sigma, t0 and the number of interacting individuals.
Reducing one or more of these values will eventually bring the graph within your computer's capacity.

[2] I will not go through specific examples, but there are numerous places that the program can be made more efficient.  This
will be a focus of future work.  For example, if the number of mutations along edges in a graph are always <20, then the program over samples random uniform deviates.

[3] The focus of program development was to get a working prototype that is accurate.  Nevertheless, one needs to be careful because
GPUs are less tolerant than CPUs to computational overload.  If a GPU is overloaded by a computation or the combination of data and
kernel in its working memory, it sometimes truncates or skips computations, without notice.  

My recommendation is to run a parameter set with a know outcome and that places a known load on the GPU.  If the results of this
are accurate, then you can have confidence that the load on the GPU is acceptable.

****************************

Output files:

Geno.out - Consists of the genotypes of sampled individuals.  Genotypes are composed of L regions consisting of 15 nucleotides.  The genotype of a region is encoded as an integer that can then be converted to a bit string (see the function bits() in the analysis file OTU-DemarkProperties-Parallel.ipynb, below).  For a single replicate the integer representation of each region is contiguously listed across individuals.  To get the genotype of an individual, partition the list into sets of length L and then convert to bits.  The function get_geno_bits() does this in the corresponding ipython notebooks (see below).

Data.out - Consists of the phenotype of every 100th individual in the graph across replicates.  Used to generate figures such as the top row in fig. 4 of Griswold (in review).

Pheno.out - Consists of the phenotypes of individuals in the present-day random sample of individuals from the community.

****************************

Example parameter files and corresponding ipython notebooks that were used to identify OTUs and other analyses are provided in the folder "example_parameter_files".  Two examples are provided.  One example corresponds to no diversifying selection nor background selection, and when sigma = 0.  GPU parameters are for a Radeon RX580.  The second example corresponds to when diversifying selection occurs and when alpha = 50.  GPU parameters are for a Nvidia Tesla P100.

The ipython notebooks do the following:

GeneralStats.ipynb - Used to plot trajectories of phenotypes in a graph and/or the histogram of phenotypes, as well as to compare phenotypic variance to neutral expectations.

OTU-DemarkProperties-Parallel.ipynb - Finds OTUs based on the 3% divergence rule and calculates population genetic and OTU abundance statistics.

OTU-CheckPi.ipynb - Used to calculate nucleotide diversity to compare with neutral expectations under the infinite sites model.

OTU-PhenoCorresp.ipynb - Also finds OTUs using the 3% rule and relates OTUs found genetically to phenotypic differences.  In addition, calculates the joint frequency spectrum and its SVD.

****************************

The program uses the Mersenne Twister random number generator of Takuji Nishimura and Makoto Matsumoto (2002) for the serial generation of the ancestral graph, and the mwc64x random number generator of DA Thomas (2011) for GPU-associated processing of the graph.  Please refer to commented files for copyright notices.
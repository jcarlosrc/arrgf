#include <stdio.h>

#define NUM_BLOCKS_X 2	//Number of cuda blocks
#define NUM_THREADS_X 16	//Number of threads per block
#define NUM_BLOCKS_Y 3	//Number of cuda blocks
#define NUM_THREADS_Y 8	//Number of threads per block

__global__ void hello(){
	printf("Hellow world, I'm thread %i %i in block %i %i\n",threadIdx.x, threadIdx.y,blockIdx.x, blockIdx.y);
}

int main(int argc, char** args){
	
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	printf("There are %lu bytes available of %lu\n", free, total);
	//Launch kernel
	hello<<<dim3(NUM_BLOCKS_X, NUM_BLOCKS_Y), dim3(NUM_THREADS_X, NUM_THREADS_X)>>>();
	//Force hellos to flush ??
	cudaDeviceSynchronize();
	
	printf("That is all!\n");
	return 0;
}

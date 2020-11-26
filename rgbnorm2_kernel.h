// This implements an l2 norm between two images. 
template<typename type> __global__ void rgbnorm2_kernel(type* f, type* g, float* out, int H, int W, int nchannels )
{
	
	//Position of block in image
	const int threadPos = blockIdx.y * blockDim.y + threadIdx.y * W + blockIdx.x * blockDim.x + threadIdx.x;
	//int bi = blockIdx.y * blockDim.y + threadIdx.y;
	//int bj = blockIdx.x * blockDim.x + threadIdx.x;
	
	float norm2 = 0.0f;
	
	for (int c = 0; c < nchannels; c++)
	{
		float temp = f[threadPos + c * H * W] - g[threadPos + c * H * W];
		temp *= temp;
		norm2 += temp;
	}
	*out += norm2;

}

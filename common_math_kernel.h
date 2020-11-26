template<typename type> __global__ void set_const_kernel(type *a, int H, int W, type val){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	//printf("blockidx = %i, blockidy = %i, i = %i, j = %j\n", blockIdx.x, blockIdx.y, i, j);
	if(i < H && j < W)
		a[i * W + j] = val;
}

template<typename type> __global__ void scale_kernel(type *a, type *out, type scale, int H, int W)
{
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		out[k] = scale * a[k];

	}
}

template<typename type> __global__ void scale_in_place_kernel(type *a, type scale, int H, int W)
{
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		a[k] = scale * a[k];

	}
}

template<typename type> __global__ void sqrt_kernel(type *a, type *out, int H, int W)
{
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		out[k] = sqrt(abs(a[k]));

	}
}

template<typename type> __global__ void sqrt_in_place_kernel(type *a, int H, int W)
{
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		a[k] = sqrt(abs(a[k]));

	}
}

template<typename type> __global__ void abs_kernel(type *a, type* out, int H, int W){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		type temp = a[k];
		if(temp < 0) temp = -temp;
		out[k] = temp;
	}
}

template<typename type> __global__ void subs_kernel(type *a_single , type *b_single, type *out_single, int H, int W){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	if(i < H && j < W)
		out_single[k] = a_single[k] - b_single[k];
}

template<typename type> __global__ void subs_kernel(type *a_single , type *b_single, int H, int W){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	if(i < H && j < W)
		a_single[k] = a_single[k] - b_single[k];
}
// To be used with drrgf-rgb, to calculate stdv using rgb metric
template<typename type> __global__ void dist_kernel_rgb(type *a , type *b, type *out_single, int nch, int H, int W){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		type sum = 0;
		for(int c = 0; c < nch; c++)
		{
			type temp = a[k + c * H * W] - b[k + c * H * W];
			sum += temp * temp;
		}	
		
		out_single[k] = sqrt(sum);
	}
}

// To be used with drrgf-rgb, to calculate stdv using rgb metric
template<typename type> __global__ void dist2_rgb_kernel(type *a , type *b, type *out_single, int H, int W, int nch){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	
	if(i < H && j < W)
	{
		type sum = 0;
		for(int c = 0; c < nch; c++)
		{
			type temp = a[k + c * H * W] - b[k + c * H * W];
			sum += temp * temp;
		}	
		
		out_single[k] = sum;
	}
}
template<typename type> __global__ void add_kernel(type *a_single , type* b_single, type* out_single, int H, int W){
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	int k = i * W + j;
	if(i < H && j < W)
		out_single[k] = a_single[k] + b_single[k];
}

//Caculate an L2 norm over regions centered at each pixel
template<typename type> __global__ void l2loc2_kernel_rgb(type* f, type* g, type* out, int H, int W, int nchannels, int kh, int kw){
	
	//Position of block in image
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	//Calculate pixel values
	if(i < H && j < W)
	{
		type norm_loc2_ij = 0;
		//Region 
		type cont = 0.0;
		for(int di = -kh; di <= kh; di++)
		{
			for(int dj = -kw; dj <= kw; dj++)
			{
				cont += 1.0;
				// extend image enough to fit posibly large kh and kw
				int e = 2;//2 * max((int) (W/kw) + 1, (int) (H / kh) + 1);
				//Extension to image positions
				int a = (di + i + e * 2*H)%(2*H) ;
				int b = (dj + j + e * 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				type norm_p = 0;
				for (int ch = 0; ch < nchannels; ch++){
					type temp =  (type)f[a * W + b + ch * nchannels] - (type)g[a * W + b + ch * nchannels];
					temp *= temp;
					norm_p += temp;
				}
				norm_loc2_ij += norm_p;			
			}		
		}
		out[i * W + j] = norm_loc2_ij / cont;
	}
}

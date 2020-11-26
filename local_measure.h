
template<typename type> __global__ void local_stdev_kernel(type* f, type* out, int H, int W, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;

	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( bi < H && bj < W)
	{
	
		type ex1 = 0;
		type ftemp = 0;
		type sum = 0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				type dist = (type)(di * di + dj * dj);
				type temp = 1.0f;
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;

				switch(weights){
				case 2:
					temp *= exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				default:
					break;
				}
				ftemp = temp * (type)f[(a) * W + b];
				ex1 += ftemp;
				sum +=temp;
			}
		}
		ex1 = ex1/sum;
		type ex2 = 0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				type temp = 1.0f;
				type dist = (type)(di * di + dj * dj); 
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				switch(weights){
				case 2:
					temp *= exp(-0.5 * ssinv2 * (di * di + dj * dj));
					break;
				case 1:
					if(dist > ss * ss) temp = 0.0f;
					break;
				default:
					break;
				}
				ftemp = ( (type)f[(a) * W + b] - ex1 );
				ex2 += temp * ftemp * ftemp;
			}
		}
		
		ex2 = ex2 /sum;
		out[bi*W + bj] = sqrt(ex2);
		
	}	

}


template<typename type> __global__ void local_stdev_kernel_rgb(type* f, type* out, int H, int W, int nch, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;

	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( bi < H && bj < W)
	{
		type global_sum = 0;
		for(int ch = 0; ch < nch; ch ++)
		{
		
			type ex1 = 0;
			type ftemp = 0;
			type sum = 0;
			for(int di = -kh; di <= kh; di ++)
			{
				for(int dj = -kw; dj <= kw; dj ++)
				{
					type dist = (type)(di * di + dj * dj);
					type temp = 1.0f;
					//Extension to image positions
					int a = (di + bi + 2*H)%(2*H) ;
					int b = (dj + bj + 2*W)%(2*W) ;
					
					if(a > H-1) a = 2*H-a-1;
					if(b > W-1) b = 2*W-b-1;

					switch(weights){
					case 2:
						temp *= exp(-0.5 * ssinv2 * dist);
						break;
					case 1:
						if(dist > ss * ss) temp = 0;
						break;
					default:
						break;
					}
					ftemp = temp * (type)f[(a) * W + b];
					ex1 += ftemp;
					sum +=temp;
				}
			}
			ex1 = ex1/sum;
			type ex2 = 0;
			for(int di = -kh; di <= kh; di ++)
			{
				for(int dj = -kw; dj <= kw; dj ++)
				{
					type temp = 1.0f;
					type dist = (type)(di * di + dj * dj); 
					//Extension to image positions
					int a = (di + bi + 2*H)%(2*H) ;
					int b = (dj + bj + 2*W)%(2*W) ;
					
					if(a > H-1) a = 2*H-a-1;
					if(b > W-1) b = 2*W-b-1;
					
					switch(weights){
					case 2:
						temp *= exp(-0.5 * ssinv2 * (di * di + dj * dj));
						break;
					case 1:
						if(dist > ss * ss) temp = 0.0f;
						break;
					default:
						break;
					}
					ftemp = ( (type)f[(a) * W + b] - ex1 );
					ex2 += temp * ftemp * ftemp;
				}
			}
			
			ex2 = ex2/sum;
			global_sum += ex2;
			
			
		
		}
		out[bi*W + bj] = sqrt(global_sum) / nch;

		
	}	

}

// Calculates the mean of a single channel image over a region using weights if wanted
template<typename type> __global__ void local_mean_kernel(type* f, type* out, int H, int W, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( bi < H && bj < W)
	{
	
		type ex1 = 0;
		type ftemp = 0;
		type sum = 0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				type dist = (type)(di * di + dj * dj);
				type temp = 1.0f;
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;

				switch(weights){
				case 2:
					temp *= exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				default:
					break;
				}
				ftemp = temp * ( f[(a) * W + b] );
				ex1 += ftemp;
				sum +=temp;
			}
		}
		ex1 = ex1/sum;
		out[bi*W + bj] = ex1;
		
	}	

}

// Calculates the mean of max_{j in V(i)} |x(j) - center_i)^2|
template<typename type> __global__ void max_dist_kernel(type* f, type* out_single, int H, int W, int nchannels, type ss, int weights, int kh, int kw){
	//Position of block in image
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( i < H && j < W)
	{
	
		type max_value = 0;
		//type max_t = 1.0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				
				//Extension to image positions
				int a = (di + i + 4*H)%(2*H) ;
				int b = (dj + j + 4*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				type norm = 0;
				type temp = 1.0;
				if(nchannels > 1)
				{
					for(int ch = 0; ch < nchannels; ch++)
					{
						temp = f[a * W + b] ;
						temp *= temp;
						norm += temp; 
					}
					norm = sqrt(norm / nchannels);
				}
				else
				{
					norm = abs(f[a * W + b]);
				}
				
				temp = 1.0;
				type dist = (type)(di * di + dj * dj);
				switch(weights){
				case 2:
					temp = exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				case 0:
					break;
				default:
					break;
				}
				norm *= temp;
				if(norm > max_value)
				{
					max_value = norm;
					//max_t = temp;
				}
			}
		}
		//max_value /= max_t;
		//if(weights == 2) max_value /= (sqrt(2.0 * 3.1416) * ss);
		out_single[ i * W + j ] = max_value;
		
	}	

}

// Calculates the mean of max_{j in V(i)} |x(j) - center_i)^2|
template<typename type> __global__ void centered_max_dist_kernel(type* f, type* center, type* out_single, int H, int W, int nchannels, type ss, int weights, int kh, int kw){
	//Position of block in image
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( i < H && j < W)
	{
	
		type max_value = 0;
		//type max_t = 1.0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				
				//Extension to image positions
				int a = (di + i + 4*H)%(2*H) ;
				int b = (dj + j + 4*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				type norm = 0;
				type temp = 1.0;
				if(nchannels > 1)
				{
					for(int ch = 0; ch < nchannels; ch++)
					{
						temp = f[a * W + b] - center[a * W + b];
						temp *= temp;
						norm += temp; 
					}
					norm = sqrt(norm / nchannels);
				}
				else
				{
					norm = abs(f[a * W + b] - center [a * W + b]);
				}
				
				temp = 1.0;
				type dist = (type)(di * di + dj * dj);
				switch(weights){
				case 2:
					temp = exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				case 0:
					break;
				default:
					break;
				}
				norm *= temp;
				if(norm > max_value)
				{
					max_value = norm;
					//max_t = temp;
				}
			}
		}
		//max_value /= max_t;
		//if(weights == 2) max_value /= (sqrt(2.0 * 3.1416) * ss);
		out_single[ i * W + j ] = max_value;
		
	}	

}



// Calculates the mean of max_{j in V(i)} |x(j) - center_i)^2|
template<typename type> __global__ void mean_dist_kernel(type* f, type* out_single, int H, int W, int nchannels, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( bi < H && bj < W)
	{
	
		type norm = 0;
		type mean_value = 0;
		type mean_weight = 0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				type dist = (type)(di * di + dj * dj);
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				norm = 0;
				type temp = 1.0;
				for(int ch = 0; ch < nchannels; ch++)
				{
					temp = f[a * W + b];
					temp *= temp;
					norm += temp; 
				}
				norm = sqrt(norm / nchannels);
				temp = (type)1;
				switch(weights){
				case 2:
					temp = exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				default:
					break;
				}
				norm *= temp;
				mean_value += norm;
				mean_weight += temp;
			}
		}
		if(mean_weight > 0) mean_value /= mean_weight;
		out_single[bi*W + bj] = mean_value;
		
		
	}	

}

// Calculates the mean of max_{j in V(i)} |x(j) - center_i)^2|
template<typename type> __global__ void centered_mean_dist_kernel(type* f, type* center, type* out_single, int H, int W, int nchannels, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( bi < H && bj < W)
	{
	
		type norm = 0;
		type mean_value = 0;
		type mean_weight = 0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				type dist = (type)(di * di + dj * dj);
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				norm = 0;
				type temp = 1.0;
				for(int ch = 0; ch < nchannels; ch++)
				{
					temp = f[a * W + b] - center[a * W + b];
					temp *= temp;
					norm += temp; 
				}
				norm = sqrt(norm / nchannels);
				temp = (type)1;
				switch(weights){
				case 2:
					temp = exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				default:
					break;
				}
				norm *= temp;
				mean_value += norm;
				mean_weight += temp;
			}
		}
		if(mean_weight > 0) mean_value /= mean_weight;
		out_single[bi*W + bj] = mean_value;
		
		
	}	

}

// Calculates the mean of sum_{j in V} (x(j) - center_i)^2 / |V|, V = V(i)
template<typename type> __global__ void squared_centered_mean_kernel(type* f, type* center, type* out, int H, int W, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;

	if( bi < H && bj < W)
	{
	
		type ex1 = 0;
		type ftemp = 0;
		type sum = 0;
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				type dist = (type)(di * di + dj * dj);
				type temp = 1.0f;
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;

				switch(weights){
				case 2:
					temp *= exp(-0.5 * ssinv2 * dist);
					break;
				case 1:
					if(dist > ss * ss) temp = 0;
					break;
				default:
					break;
				}
				ftemp =  f[a * W + b] - center[a * W + b];
				ftemp = temp * ftemp * ftemp;
				ex1 += ftemp;
				sum +=temp;
			}
		}
		out[bi*W + bj] = ex1 / sum;
		
	}	

}

template<typename type> __global__ void sum_channels_kernel(type* f_rgb, type* out_single, int H, int W, int nchannels){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(bi < H && bj < W)
	{
		type sum = 0;
		for(int ch = 0; ch < nchannels; ch++)
		{
			sum += f_rgb[ch * H * W + bi * W + bj];
		}
		
		out_single[bi * W + bj] = sum;
	}
}

template<typename type> __global__ void local_max_kernel(type* f, type* out, int H, int W, type ss, int weights, int kh, int kw){
	//Position of block in image
	int bj = blockIdx.x * blockDim.x + threadIdx.x ;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	
	type ssinv2 = 1.0f / ss;
	ssinv2 *= ssinv2;
	type local_max = 0.0f;

	if( bi < H && bj < W)
	{	
		for(int di = -kh; di <= kh; di ++)
		{
			for(int dj = -kw; dj <= kw; dj ++)
			{
				//type dist = (type)(di * di + dj * dj);
				//Extension to image positions
				int a = (di + bi + 2*H)%(2*H) ;
				int b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				//type tempmax = 0.0f;
				//type temp = 1.0f;
				
				// No weights for max yet
				/*switch(weights){
				case 1:
					temp *= exp(-0.5 * ssinv2 * dist);
					break;
				case 0:
					if(dist > ss * ss) temp = 0;
					break;
				default:
					break;
				}
				*/
				
				local_max = max(local_max, f[a * W + b]);
			}
		}
		
		out[(bi)*W + bj] = local_max;
		
	}

}

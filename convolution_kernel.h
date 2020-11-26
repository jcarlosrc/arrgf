//NOT shared memory for different kernels
template<typename type> __global__ void convolution_kernel(type* f, type* out, int H, int W, type s, int ker, int kh, int kw){
	//Position of block in image
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	type sinv2 = 1.0f / s;
	sinv2 *= sinv2;
	//Calculate pixel values
	int a, b;
	type num = 0; type den = 1.0;
	if(bi < H && bj < W){
		num = 0; 
		den = 0;
		for(int di = -kh; di<=kh; di++){
			for(int dj = -kw; dj<=kw; dj++){
				type temp = 1.0f;
				type dist2 = (type)(di * di + dj * dj);
				switch(ker){
				case 0:				
					temp *= exp(-0.5 * dist2 * sinv2);
					break;
				case 2:
					if(dist2 > s * s) temp = 0.0f;
					break;
				default:
					break;
				}	
				//Extension to image positions
				a = (di + bi + 2*H)%(2*H) ;
				b = (dj + bj + 2*W)%(2*W) ;
				
				if(a > H-1) a = 2*H-a-1;
				if(b > W-1) b = 2*W-b-1;
				
				num += temp * f[a * W + b];
				den += temp;
			}		
		}
	
		out [(bi) * W + bj] = num / den;
	}
}

//Convolution kernel, for regularization, using shared memory, NOT RELIABLE

template<typename type, int kh, int kw, int sharedSizeH, int sharedSizeW> __global__ void convolution_kernel_gaussian(type* f, type* out, int H, int W, float sigma, int nRecH){
	int bi = nRecH * blockIdx.y * blockDim.y;
	int bj =  1 * blockIdx.x * blockDim.x;

	__shared__ type fE[sharedSizeH * sharedSizeW];

	// Load part of f to share memory
	for(int ii = threadIdx.y; ii < sharedSizeH-blockDim.y; ii+= blockDim.y){
		for(int jj = threadIdx.x; jj < sharedSizeW; jj+= blockDim.x){
			//Corresponding position in image
			int a = (ii - kh + bi + 2*H) % (2*H);
			int b = (jj - kw + bj + 2*W) % (2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			fE[(ii)*(sharedSizeW) + jj] = *(f + (a)*W + b);
		}
	}

	int L = sharedSizeH - blockDim.y;
	//Start loop to load blocks of BH height to same shared memory
	for(int k = 0; k < nRecH; k++){
	
		int ii = threadIdx.y;
		//Load Inferior part of BH x EH elements and save it to proper positions in fE
		for(int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (L + ii - kh + bi + 2*H)%(2*H);
			int b = (jj - kw + bj + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			fE[((L + ii + sharedSizeH) % sharedSizeH) * sharedSizeW + jj] = *(f + (a)*W + b);
		}
		L = L + blockDim.y;
		__syncthreads();

		//Calculate pixel values
		int i = threadIdx.y;
		int j = threadIdx.x;
		if(bi+i+k*blockDim.y < H && bj+j < W){
			float num = 0;
			float den = 0;
			float temp = 0;
			for(int di = -kh; di <= kh; di++){
				for(int dj = -kw; dj <= kw; dj++){
					//Convolution Kernel
					temp = exp(-(float)(di*di + dj*dj)/(2.0f*sigma*sigma));
					num += ((float)(fE[( (kh+i+di+k*blockDim.y + sharedSizeH) % sharedSizeH) * sharedSizeW +kw+j+dj]))  * temp;
					den += temp;
				}
			}
		*(out + (bi+i+k*blockDim.y)*W + bj+j) = (type)(num/den);
		}
		__syncthreads();
	}

}



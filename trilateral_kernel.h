//type infval = 0.0010;
#define PI 3.141592;
#define macro_get_val(f, i, j, c, H, W)	((i > -1 && i < H && j > -1 && j < W) ? f[i * W + j + c * H * W] : 0.0f)

//Kernel with RGB distance calculation, NOT shared memory, vector sr. f is single channel, g can be rgb, out is single channel
template<typename type> __global__ void trilateral_kernel_rgb(type* f, type* g, type* out, int nchannels, int H, int W, int domain_extension, type ss, type sr, type* srvec, type infsr, type sm, int stype, int itype, int mtype, int kh, int kw){
	//Position of block in image
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	int bj = blockIdx.x * blockDim.x + threadIdx.x;

	//Calculate pixel values
	if(bi < H && bj < W){
		
		type auxsr = srvec[(bi) * W + bj];

		// Typical adaptive algorithm
		auxsr = max(infsr, sr * auxsr);
				
		// Alternative adaptive algorithm
		//auxsr = sr / max(0.001f, auxsr);
		
		//if(auxsr < infsr)
		//{
		//	auxsr = 0.001f  + 1.0f * ( 1.0f  -  auxsr / infsr);
		//}
		//auxsr = sr * auxsr;
		//type auxsr = sr / max(sqrt(srvec[bi * W + bj]), 0.001);

		// if(auxsr < 0.01) auxsr = 1.0;
		//auxsr = max(infsr, auxsr);
		//auxsr = max(0.01, sr * srvec[(bi) * W + bj]);
		//type auxsm = max(infval, sm * srvec[(bi) * W + bj]);
		type auxsm = sm;
		type num = 0; 
		type den = 0;
		for(int di = -kh; di<=kh; di++){
			for(int dj = -kw; dj<=kw; dj++){
				int a, b, e;

				switch (domain_extension)
				{
				case 0:
					a = bi + di; b = bj + dj;
					break;
				case 1:
					e = 2 * max((int) (W/kw) + 1, (int) (H / kh) + 1);
					a = (di + bi + e * 2*H)%(2*H) ;
					b = (dj + bj + e * 2*W)%(2*W) ;
					
					if(a > H-1) a = 2*H-a-1;
					if(b > W-1) b = 2*W-b-1;
					break;
				default:
					a = bi + di; b = bj + dj;
					break;
				}
				
				type temp = 1.0f;
				// Apply spatial kernel
				if(stype > -1 && ss>0)
				{
					// Radius of desired ball centered at each pixel
					type dist2 = (type)(di * di + dj * dj);
					switch (stype){
					case 0:	// gaussian
						temp *= exp(-0.5f * dist2 /ss/ss);
						break;
					case 1:	// tukey
						temp *= max(0.0f, 1.0f-dist2 / ss /ss) * max(0.0f, 1.0f-dist2 / ss /ss);
						break;
					case 2:	//Circle
						if(ss * ss < dist2) temp = 0;
						break;
					case 3:	// Lorentz
						temp *= 1.0f / (1.0f + dist2 / ss / ss);
						break;
					case 4:	// Hamming-windowed sinc
						if(ss * abs(di)> 0)
						{
							temp *= sin(3.14159265 * ss * di) * ( 0.42f - 0.5f*cos(2.0f * 3.14159265 * di * ss / (2.0 * kh) + 3.14159265) +  0.08f*cos(4.0f * 3.14159265 * di * ss / (2.0 * kh) + 2.0f * 3.14159265) ) / ( 3.14159265 * ss * di);					
						}
						else
						{
							temp *= 0.42f - 0.5f * cos(2.0f * 3.14159265f * di * ss / (2.0 * kh) + 3.14159265f) + 0.08f * cos(4.0f * 3.14159265 * di * ss / (2.0 * kh) + 2.0f * 3.14159265);
						}

						if(ss * abs(dj)> 0)
						{
							temp *= sin(3.14159265 * ss * dj) * ( 0.42f - 0.5f*cos(2.0f * 3.14159265 * dj * ss / (2.0 * kw) + 3.14159265) +  0.08f*cos(4.0f * 3.14159265 * dj * ss / (2.0 * kw) + 2.0f * 3.14159265) ) / ( 3.14159265 * ss * dj);
						}
						else
						{
							temp *= 0.42f - 0.5f * cos(2.0f * 3.14159265f * dj * ss / (2.0 * kw) + 3.14159265f) + 0.08f * cos(4.0f * 3.14159265 * dj * ss / (2.0 * kw) + 2.0f * 3.14159265);
						}

						break;
					case 5:	// sinc
						if(di != 0)
						{
							temp *= sin(3.14159265 * ss * di) / (3.14159265 * ss * di);					
						}

						if(dj != 0)
						{
							temp *= sin(3.14159265 * ss * dj) / ( 3.14159265 * ss * dj);
						}	

						break;
					default:
						break;
					}
				}
				// Add range kernel
				if(itype >= 0 && sr > 0)
				{
					//RGB distance, sums over all channels
					type norm2 = 0;
					for(int c = 0; c < nchannels; c++)
					{
						type temp = macro_get_val(g, a, b, c, H, W) - macro_get_val(g, bi, bj, c, H, W);
						//type temp =  g[a * W + b + c * H * W] - g[(bi) * W + bj + c * H * W];
						norm2+= temp * temp;
					}
					//norm2 = norm2 / ((type)nchannels * (type) nchannels);
					norm2 = norm2 / ((type) nchannels );
					switch (itype){
					case 0:
						temp *= exp(-0.5f * norm2 /auxsr/auxsr);
						break;
					case 1:
						temp *= max(0.0f, 1.0f-norm2 / auxsr /auxsr) * max(0.0f, 1.0f-norm2 / auxsr /auxsr);
						break;
					case 2:
						if(auxsr * auxsr < norm2)
							temp = 0.0f;
						break;
					case 3:
						temp *= 1.0f / (1.0f + norm2 / auxsr / auxsr);
						break;
					default:
						break;
					}
				}
				
				
				//RGB distance, sums over all channels
				if(mtype >= 0 && sm > 0){
				
					type norm2 = 0;
					for(int c = 0; c < nchannels; c++)
					{
						type temp = macro_get_val(f,a, b, 0, H, W) - macro_get_val(g, a, b, c , H, W);
						//type temp =  f[a * W + b + c * H * W] - g[a * W + b + c * H * W];
						norm2 += temp * temp;
					}
					norm2 = norm2 / ((type)nchannels);
					
					switch (mtype){
					case 0:
						temp *= exp(-0.5f  * norm2 /auxsm/auxsm );
						break;
					case 1:
						temp *= max(0.0f, 1.0f- norm2 /auxsm/auxsm) * max(0.0f, 1.0f - norm2 / auxsm/auxsm);
						break;
					case 2:
						if(sm * sm < norm2) temp = 0.0f;
						break;
					case 3:
						temp *= 1.0f / (1.0f + norm2 / auxsm / auxsm);
						break;
					default:
						break;
					}
				}
				
				num += temp * macro_get_val(f, a, b, 0, H, W);
				den += temp;
			}		
		}
	
		out [(bi) * W + bj] = num / den;
	}
}

//Kernel with RGB distance calculation, NOT shared memory, single sr
template<typename type> __global__ void trilateral_kernel_rgb(type* f, type* g, type* out, int nchannels, int H, int W, int domain_extension, type ss, type sr, type sm, int stype, int itype, int mtype, int kh, int kw){
	
	//Position of block in image
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	int bj = blockIdx.x * blockDim.x + threadIdx.x;
	
	type ss2inv = 1.0f / ss/ss;

	//Calculate pixel values
	if(bi < H && bj < W){
		type num = 0; 
		type den = 0;
		for(int di = -kh; di <= kh; di++){
			for(int dj = -kw; dj <= kw; dj++){
				// extend image enough to fit posibly large kh and kw
				//2 * max((int) (W/kw) + 1, (int) (H / kh) + 1);
				//Extension to image positions
				int a, b, e;
				switch (domain_extension)
				{
				case 0:
					a = bi + di;
					b = bj + dj;
					break;
				case 1:
					e = 2 * max((int) (W/kw) + 1, (int) (H / kh) + 1);
					a = (di + bi + e * 2*H)%(2*H) ;
					b = (dj + bj + e * 2*W)%(2*W) ;
					
					if(a > H-1) a = 2*H-a-1;
					if(b > W-1) b = 2*W-b-1;
					break;
				default:
					a = bi + di;
					b = bj + dj;
					break;
				}
				
				type temp = 1.0f;
				// Apply spatial kernel
				if(stype > -1 && ss>0)
				{
					type dist2= (type)(di * di + dj * dj);
					switch (stype){
					case 0:	// gaussian
						temp *= exp(-0.5f * dist2* ss2inv);
						//temp *= 1.0f;
						break;
					case 1:	// tukey
						temp *= max(0.0f, 1.0f-dist2/ ss /ss) * max(0.0f, 1.0f-dist2/ ss /ss);
						break;
					case 2:
						if(ss * ss < dist2) temp = 0;
						break;
					case 3:
						temp *= 1.0f / (1.0f + dist2/ ss / ss);
						break;
					case 4:	// Hamming-windowed sinc
						if(ss * abs(di)> 0)
						{
							temp *= sin(3.14159265 * ss * di) * ( 0.42f - 0.5f*cos(2.0f * 3.14159265 * di * ss / (2.0 * kh) + 3.14159265) +  0.08f*cos(4.0f * 3.14159265 * di * ss / (2.0 * kh) + 2.0f * 3.14159265) ) / ( 3.14159265 * ss * di);					
						}
						else
						{
							temp *= 0.42f - 0.5f * cos(2.0f * 3.14159265f * di * ss / (2.0 * kh) + 3.14159265f) + 0.08f * cos(4.0f * 3.14159265 * di * ss / (2.0 * kh) + 2.0f * 3.14159265);
						}

						if(ss * abs(dj)> 0)
						{
							temp *= sin(3.14159265 * ss * dj) * ( 0.42f - 0.5f*cos(2.0f * 3.14159265 * dj * ss / (2.0 * kw) + 3.14159265) +  0.08f*cos(4.0f * 3.14159265 * dj * ss / (2.0 * kw) + 2.0f * 3.14159265) ) / ( 3.14159265 * ss * dj);
						}
						else
						{
							temp *= 0.42f - 0.5f * cos(2.0f * 3.14159265f * dj * ss / (2.0 * kw) + 3.14159265f) + 0.08f * cos(4.0f * 3.14159265 * dj * ss / (2.0 * kw) + 2.0f * 3.14159265);
						}

						break;
					case 5:	// sinc = sinc(di) * sinc(dj)
						if(di != 0)
						{
							temp *= sin(3.14159265 * ss * di) / (3.14159265 * ss * di);					
						}
						
						if(dj != 0)
						{
							temp *= sin(3.14159265 * ss * dj) / (3.14159265 * ss * dj);
						}

						break;
					default:
						break;
					}
				}

				
				if(itype >= 0 && sr > 0)
				{
					//RGB distance, sums over all channels
					type norm = 0;
					for(int c = 0; c < nchannels; c++)
					{
						//type temp =  g[a * W + b + c * H * W] - g[(bi) * W + bj + c * H * W];
						type temp = macro_get_val(g, a, b, c, H, W) - macro_get_val(g, bi, bj, c, H, W);
						norm += temp * temp;
					}
					norm = norm / ((type)nchannels);
					
					switch (itype){
					case 0:
						temp *= exp(-0.5f * norm /sr/sr);
						break;
					case 1:
						temp *= max(0.0f, 1.0f-norm/ sr /sr) * max(0.0f, 1.0f-norm / sr /sr);
						break;
					case 2:
						if(sr * sr < norm)
							temp = 0.0f;
						break;
					case 3:
						temp *= 1.0f / (1.0f + norm / sr / sr);
						break;
					default:
						break;
					}
				}
				
				
				//RGB distance, sums over all channels
				if(mtype >= 0 && sm > 0)
				{
					type norm = 0;
					for(int c = 0; c < nchannels; c++)
					{
						//type temp =  f[a * W + b] - g[a * W + b + c * H * W];
						type temp = macro_get_val(f, a, b, 0, H, W) - macro_get_val(g, a, b, c, H, W);
						norm += temp * temp;
					}
					norm = norm / ((type)nchannels);
					
					switch (mtype){
					case 0:
						temp *= exp(-0.5f  * norm /sm/sm );
						break;
					case 1:
						temp *= max(0.0f, 1.0f- norm/sm/sm) * max(0.0f, 1.0f - norm / sm/sm);
						break;
					case 2:
						if(sm * sm < norm) temp = 0.0f;
						break;
					case 3:
						temp *= 1.0f / (1.0f + norm / sm / sm);
						break;
					default:
						break;
					}
				}
				
				//num += temp * f[a * W + b];
				num += temp * macro_get_val(f, a, b, 0, H, W);
				den += temp;
			}		
		}

		out [(bi) * W + bj] = num / den;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////// //                       Using shared memory, not ready.              ///////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Bilateral kernel.

template<typename type, int kh, int kw, int sharedSizeH, int sharedSizeW> __global__ void trilateral_kernel(type* f, type* g, type* out, int H, int W, type sigma_r, type sigma_s, int nRecH){
	//Position of block in image
	int bi = nRecH * blockIdx.y * blockDim.y;
	int bj =  1 * blockIdx.x * blockDim.x;

	__shared__ type fE[sharedSizeH * sharedSizeW];
	__shared__ type gE[sharedSizeH * sharedSizeW];

	// Load part of f to share memory
	for(int ii = threadIdx.y; ii < sharedSizeH-blockDim.y; ii+= blockDim.y){
		for(int jj = threadIdx.x; jj < sharedSizeW; jj+= blockDim.x){
			//Corresponding position in image
			int a = (ii - kh + bi + 2*H)%(2*H);
			int b = (jj - kw + bj + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			fE[(ii)*(sharedSizeW) + jj] = *(f + (a)*W + b);
		}
	}
	
	//Same for g
	for (int ii = threadIdx.y; ii < sharedSizeH-blockDim.y; ii += blockDim.y){
		for (int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (ii + bi-kh + 2*H)%(2*H);
			int b = (jj + bj-kw + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			gE[(ii)*(sharedSizeW) + jj] = *(g + (a)*W + b);
		}
	}
	int L = sharedSizeH - blockDim.y;

	//Start loop to load blocks of blockDim.y height to same shared memory
	for(int k = 0; k < nRecH; k++){
		int ii = threadIdx.y;
		//Load Inferior part of blockDim.y x sharedSizeH elements and save it to proper positions in fE
		for(int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (L + ii - kh + bi + 2*H)%(2*H);
			int b = (jj - kw + bj + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			fE[((L + ii + sharedSizeH)%sharedSizeH) * sharedSizeW + jj] = *(f + (a)*W + b);
		}
		//Same for g
		for(int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (L + ii + bi-kh + 2*H)%(2*H);
			int b = (jj + bj-kw + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			gE[((L + ii + sharedSizeH)%sharedSizeH) * sharedSizeW + jj] = *(g + (a)*W + b);
		}
		
		L = L + blockDim.y;
		__syncthreads();

		//Calculate pixel values
		int i = threadIdx.y;
		int j = threadIdx.x;
		if(bi+i+k*blockDim.y < H && bj+j < W){
			type num = 0; 
			type den = 0;
			int count = 0;
			for(int di = -kh; di<=kh; di++){
				for(int dj = -kw; dj<=kw; dj++){
					type temp = exp(-(type)(di*di + dj*dj)/(2.0f*sigma_s*sigma_s));
					type temp1 = (type)(gE[ ((kh+i+di+k*blockDim.y + sharedSizeH) % sharedSizeH) * sharedSizeW + kw+j+dj ] - gE[((kh+i+k*blockDim.y + sharedSizeH)%sharedSizeH)*sharedSizeW + kw+j])/sigma_r;
					temp1 = exp(- 0.5 * temp1 * temp1);
					temp *= temp1;
					//temp =  exp( -(pow( (type)(gE[ ((kh+i+di+k*blockDim.y + sharedSizeH) % sharedSizeH) * sharedSizeW + kw+j+dj ])/CTE255 - ((type)(gE[((kh+i+k*blockDim.y + sharedSizeH)%sharedSizeH)*sharedSizeW + kw+j]))/CTE255 , 2.0f)) / (2.0f*sigma_r * sigma_r) ) * temp;
					num += ((type)(fE[ ((kh+i+di+k*blockDim.y) % sharedSizeH) * sharedSizeW +kw+j+dj ])) * temp;
					den += temp;
					count ++;
				}
			}
			*(out + (bi+i+k*blockDim.y)*W + bj+j) = (type) (num/den);
		}
		__syncthreads();
	}

}

// Same but vector sr
template<typename type, int kh, int kw, int sharedSizeH, int sharedSizeW> __global__ void trilateral_kernel(type* f, type* g, type* out, int H, int W, type* sigma_r, type sigma_s, int nRecH){
	//Position of block in image
	int bi = nRecH * blockIdx.y * blockDim.y;
	int bj =  1 * blockIdx.x * blockDim.x;

	__shared__ type fE[sharedSizeH * sharedSizeW];
	__shared__ type gE[sharedSizeH * sharedSizeW];

	// Load part of f to share memory
	for(int ii = threadIdx.y; ii < sharedSizeH-blockDim.y; ii+= blockDim.y){
		for(int jj = threadIdx.x; jj < sharedSizeW; jj+= blockDim.x){
			//Corresponding position in image
			int a = (ii - kh + bi + 2*H)%(2*H);
			int b = (jj - kw + bj + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			fE[(ii)*(sharedSizeW) + jj] = *(f + (a)*W + b);
		}
	}
	
	//Same for g
	for (int ii = threadIdx.y; ii < sharedSizeH-blockDim.y; ii += blockDim.y){
		for (int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (ii + bi-kh + 2*H)%(2*H);
			int b = (jj + bj-kw + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			gE[(ii)*(sharedSizeW) + jj] = *(g + (a)*W + b);
		}
	}
	int L = sharedSizeH - blockDim.y;

	//Start loop to load blocks of blockDim.y height to same shared memory
	for(int k = 0; k < nRecH; k++){
		int ii = threadIdx.y;
		//Load Inferior part of blockDim.y x sharedSizeH elements and save it to proper positions in fE
		for(int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (L + ii - kh + bi + 2*H)%(2*H);
			int b = (jj - kw + bj + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			fE[((L + ii + sharedSizeH)%sharedSizeH) * sharedSizeW + jj] = *(f + (a)*W + b);
		}
		//Same for g
		for(int jj = threadIdx.x; jj < sharedSizeW; jj += blockDim.x){
			//Corresponding position in image
			int a = (L + ii + bi-kh + 2*H)%(2*H);
			int b = (jj + bj-kw + 2*W)%(2*W);
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			gE[((L + ii + sharedSizeH)%sharedSizeH) * sharedSizeW + jj] = *(g + (a)*W + b);
		}
		
		L = L + blockDim.y;
		__syncthreads();

		//Calculate pixel values
		int i = threadIdx.y;
		int j = threadIdx.x;
		if(bi+i+k*blockDim.y < H && bj+j < W){
			type sr = sigma_r[(bi+i+k*blockDim.y) * W + bj+j];
			type num = 0; 
			type den = 0;
			int count = 0;
			for(int di = -kh; di<=kh; di++){
				for(int dj = -kw; dj<=kw; dj++){
					type temp = exp(-(type)(di*di + dj*dj)/(2.0f*sigma_s*sigma_s));
					type temp1 = (type)(gE[ ((kh+i+di+k*blockDim.y + sharedSizeH) % sharedSizeH) * sharedSizeW + kw+j+dj ] - gE[((kh+i+k*blockDim.y + sharedSizeH)%sharedSizeH)*sharedSizeW + kw+j])/sr;
					temp1 = exp(- 0.5 * temp1 * temp1);
					temp *= temp1;
					//temp =  exp( -(pow( (type)(gE[ ((kh+i+di+k*blockDim.y + sharedSizeH) % sharedSizeH) * sharedSizeW + kw+j+dj ])/CTE255 - ((type)(gE[((kh+i+k*blockDim.y + sharedSizeH)%sharedSizeH)*sharedSizeW + kw+j]))/CTE255 , 2.0f)) / (2.0f*sigma_r * sigma_r) ) * temp;
					num += ((type)(fE[ ((kh+i+di+k*blockDim.y) % sharedSizeH) * sharedSizeW +kw+j+dj ]))  * temp;
					den += temp;
					count ++;
				}
			}
			*(out + (bi+i+k*blockDim.y)*W + bj+j) = (type) (num/den);
		}
		__syncthreads();
	}

}

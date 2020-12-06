
#ifndef __MATRIX_H__
#define __MATRIX_H__

#define MATRIX_RGB_TYPE_CHUNK 0
#define MATRIX_RGB_TYPE_INTER 1

#define MATRIX_COLS 0
#define MATRIX_ROWS 1

#include "patch.h"

namespace Matrix
{	
	/* Transforms an image to gray scale */
	/* Input : RGB[3 * H * W], Output: Single [H * W]*/
	template<typename type> int RGB_to_Gray(type *in, type *out, int length, int nchannels = 3, int rgb_format = MATRIX_RGB_TYPE_INTER)
	{
		if(nchannels == 3)	// RGB images
		{
			dprintf(0, "rgb_to_gray: 3 channel input");
			switch (rgb_format)
				{
				case MATRIX_RGB_TYPE_CHUNK: /* input is supposed to be in the format R( r r r ... ) G( g g g g ... ) B( b b b b ... ) */			
					dprintf(0, " rgb_to_gray : chunk\n");
					for(int i = 0; i < length; i++)
					{
						out[i] = 0.2989 * in[i + 0] + 0.587 * in[i + 1 * length] + 0.114 * in[i + 2 * length] ;
					}
					dprintf(0, "rgb_to_gray: done\n");
					return 1;
					break;
				case MATRIX_RGB_TYPE_INTER: /* input is supposed to be in the format L1 (r, g, b, r, g, b, ...) L2 ... */
					dprintf(0, " rgb_to_gray : inter\n");
					for(int i = 0; i < length; i++)
					{
						out[i] = 0.2989 * in[i * nchannels] + 0.587 * in[i * nchannels + 1] + 0.114 * in[i * nchannels + 2] ;
					}
					return 1;
					break;
				
				default:
					return nchannels;
					break;
				}
		}
		else if(nchannels == 1)
		{
			for(int i = 0; i < length; i++)
			{
				out[i] = in[i];
			}
			return nchannels;
		}
		else
		{
			return nchannels;
		}
	}

	template<typename type> int change_rgb_format(type *input, type *output, int lenght, int nchannels, int rgb_format)
	{
		switch (rgb_format)
		{
		case MATRIX_RGB_TYPE_INTER:
			for(int i = 0; i < lenght; i++)
			{
				for(int c = 0; c < nchannels; c++)
				{
					output[i  + c * lenght] = input[nchannels * i + c ];
				}
			}
			return MATRIX_RGB_TYPE_CHUNK;
			break;
		
		case MATRIX_RGB_TYPE_CHUNK:
			for(int i = 0; i <lenght ; i++)
			{
				for(int c = 0; c < nchannels; c++)
				{
					output[ nchannels * i + c] = input[ i + c * lenght ];
				}
			}
			return MATRIX_RGB_TYPE_INTER;
		default:
			return rgb_format;
			break;
		}

	}

	/* Transposes data from cols to lines or vis. Does NOT transposes image or rgb format. Format is still RGB RGB for each pixel. Does NOT support output = input */
	template<typename type> int change_data_format(type *input, type *output, int H, int W, int nchannels, int input_data_format =  MATRIX_COLS, int rgb_format = MATRIX_RGB_TYPE_INTER)
	{
		switch (rgb_format)
		{
			case MATRIX_RGB_TYPE_INTER:
				switch (input_data_format)
				{
				case MATRIX_COLS:
					dprintf(0, "change_data_format: MATRIX_RGB_TYPE_INTER, MATRIX_COLS\n");
					for(int j = 0; j < W; j ++)
					{
						for(int i = 0; i < H; i++)
						{
							for(int channel = 0; channel < nchannels; channel++)
							{
								output[i * W * nchannels + j * nchannels + channel] = input [j * H * nchannels + i * nchannels + channel];
							}
						}
					}
					return MATRIX_ROWS;
					break;
				
				case MATRIX_ROWS:
					dprintf(0, "change_data_format: MATRIX_RGB_TYPE_INTER, MATRIX_ROWS\n");
					for(int i = 0; i < H; i ++)
					{
						for(int j = 0; j < W; j++)
						{
							for(int channel = 0; channel < nchannels; channel++)
							{
								output[j * H * nchannels + i * nchannels + channel] = input [i * W * nchannels + j * nchannels + channel];
							}
						}
					}
					return MATRIX_COLS;
					break;		

				default:
					return input_data_format;
					break;
				}
				break;
			
			default:
			return input_data_format;
				break;
		}

		return input_data_format;
	}

	template<typename type> int apply_gamma(type *in, type *out, int lenght)
	{
		for(int k = 0; k < lenght; k++)
		{
			out[k] = ( in[k] <= 0.0031308) ? 12.92 * in[k] : 1.055 * pow( in[k], 1.0/2.4 ) - 0.055;
		}
		return 1;
	}
	template<typename type> type apply_gamma(type val)
	{
		return ( val <= 0.0031308) ? 12.92 * val : 1.055 * pow( val, 1.0/2.4 ) - 0.055;
	}

	template<typename type> int ungamma(type *in, type *out, int lenght)
	{
		for(int k = 0; k < lenght; k++)
		{
			type val = in[k];
			out[k] = (val <= 0.04045) ? val = val/12.92f : pow( (val + 0.055f)/1.055, 2.4f);
		}
		return 1;
	}
	template<typename type> int ungamma(type *in, type *out, int H, int W, int nchannels, int rgb_format = MATRIX_RGB_TYPE_CHUNK)
	{
		return 1;
	}
	template<typename type> type ungamma(type val)
	{
		return (val <= 0.04045) ? val = val/12.92f : pow( (val + 0.055f)/1.055, 2.4f);
	}

	template<typename type> int my_log(type *in, type *out, int lenght, type eps = 0.0f)
	{
		for(int k = 0; k < lenght; k++)
		{
			out[k] = ((in[k] + eps > 0) ? log(in[k] + eps) : 0);
		}
		return 1;
	}
	 template<typename type> int float_to_uint(type *in, unsigned int *out, int lenght, int scale)
	{
		for(int k = 0; k < lenght; k++)
		{
			out[k] = (unsigned int)(in[k] * scale);
		}
		return 1;
	}

	// L2 norm
	template<typename type> type max_rgb_norm2(type *a, int H, int W, int nch)
	{
		type max_norm2 = 0.0f;
		for (int i = 0; i < H; i ++)
		{
			for (int j = 0; j < W; j++)
			{
				type norm_i2 = 0.0f;
				for (int ch = 0; ch < nch; ch ++)
				{
					type temp = a[i * W + j + ch * nch];
					temp *= temp;
					norm_i2 += temp;
				}
				if(max_norm2 < norm_i2/nch)
				{
					max_norm2 = norm_i2/nch;
				}
				//norm2 += norm_i2;
			}
		}
		return max_norm2;
		//return norm2 / (H * W);
	}

	// L2 norm
	template<typename type> type min_rgb_norm2(type *a, int H, int W, int nch)
	{
		type min_norm2 = 0.0f;
		for (int i = 0; i < H; i ++)
		{
			for (int j = 0; j < W; j++)
			{
				type norm_i2 = 0.0f;
				for (int ch = 0; ch < nch; ch ++)
				{
					type temp = a[i * W + j + ch * nch];
					temp *= temp;
					norm_i2 += temp;
				}
				if(min_norm2 > norm_i2/nch)
				{
					min_norm2 = norm_i2/nch;
				}
				//norm2 += norm_i2;
			}
		}
		return min_norm2;
		//return norm2 / (H * W);
	}

	template<typename type> void sub(type *a, type *b, type *out, int lenght)
	{
		for(int i = 0; i < lenght; i++)
		{
			out[i] = a[i] - b[i];
		}
	}

	template<typename type> void add(type *a, type *b, type *out, int lenght)
	{
		for(int i = 0; i < lenght; i++)
		{
			out[i] = a[i] + b[i];
		}
	}

	template<typename type> void scale(type *a, type factor, type *out, int lenght)
	{
		for(int i = 0; i < lenght; i++)
		{
			out[i] = a[i] * factor;
		}
	}
	template<typename type> void dot_mult(type *a, type *b, type *out, int lenght)
	{
		for(int i = 0; i < lenght; i++)
		{
			out[i] = a[i] * b[i];
		}
	}

	template<typename type> void dot_div(type *a, type *b, type *out, int lenght )
	{
		for(int i = 0; i < lenght; i++)
		{
			out[i] = (b[i] != 0) ? a[i] / b[i] : 0.0;
		}
	}

	template<typename type> void my_exp(type *a, type *out, int lenght, double base_log = 1.0)
	{
		for(int i = 0; i < lenght; i++)
		{
			out[i] = (type)exp(base_log * (double)a[i]);
		}
	}


};


template<typename type> type L_infty_norm2(type *a, type * b, int H, int W, int nch)
{
	type norm2 = 0.0f;
	for (int i = 0; i < H; i ++)
	{
		for (int j = 0; j < W; j++)
		{
			type norm_i2 = 0.0f;
			for (int ch = 0; ch < nch; ch ++)
			{
				type temp = a[i * W + j + ch * nch] - b[i * W + j + ch * nch];
				temp *= temp;
				norm_i2 += temp;
			}
			norm2 = max(norm2, norm_i2);
		}
		
	}
	return norm2;
}

// L2 norm
template<typename type> type L2_norm2(type *a, type * b, int H, int W, int nch)
{
	type norm2 = 0.0f;
	for (int i = 0; i < H; i ++)
	{
		for (int j = 0; j < W; j++)
		{
			type norm_i2 = 0.0f;
			for (int ch = 0; ch < nch; ch ++)
			{
				type temp = a[i * W + j + ch * nch] - b[i * W + j + ch * nch];
				temp *= temp;
				norm_i2 += temp;
			}
			norm2 += norm_i2;
		}
		
	}
	return norm2 / (H * W);
}



// max norm for single channel
template<typename type> type max_abs(type *a, int H, int W)
{
	type norm = 0.0f;
	for (int i = 0; i < H; i ++)
	{
		for (int j = 0; j < W; j++)
		{
			type temp = abs(a[i * W + j]);
			if (temp > norm)
			{
				norm = temp;
			}
		}
	}
	return norm;
}	
	
template<typename type> type max_f(type *a, int H, int W)
{
	type norm = a[0];
	for (int i = 0; i < H; i ++)
	{
		for (int j = 0; j < W; j++)
		{
			type temp = a[i * W + j];
			if (temp > norm)
			{
				norm = temp;
			}
		}
	}
	return norm;
}

template<typename type> type min_f(type *a, int H, int W)
{
	type norm = a[0];
	for (int i = 0; i < H; i ++)
	{
		for (int j = 0; j < W; j++)
		{
			type temp = a[i * W + j];
			if (temp < norm)
			{
				norm = temp;
			}
		}
	}
	return norm;
}

template <typename type> type mean(type *a, int H, int W)
{
	type mean = 0.0f;
	for (int i = 0; i < H; i ++)
	{
		for (int j = 0; j < W; j++)
		{
			mean += a[i * W + j];
		}
	}
	return mean / (H * W);
}				
				
// L2loc norm, too slow. Better in gpu
template<typename type> type L2loc_norm2(type *vec_a, type * vec_b, int H, int W, int nch, int kh, int kw)
{

	type norm2 = 0.0f;
	for (int i = 0; i < H; i ++){
	for (int j = 0; j < W; j++)
	{
		type norm_loc2 = 0.0f;
		for(int di = -kh; di<=kh; di++){
		for(int dj = -kw; dj<=kw; dj++){
			type norm_i2 = 0.0f;
			// Calculate positions in extended image
			int e = 2 * max((int) (W/kw) + 1, (int) (H / kh) + 1);
			int a = (di + i + e * 2*H)%(2*H) ;
			int b = (dj + j + e * 2*W)%(2*W) ;
	
			if(a > H-1) a = 2*H-a-1;
			if(b > W-1) b = 2*W-b-1;
			
			for (int ch = 0; ch < nch; ch ++){
				type temp = vec_a[i * W + j + ch * nch] - vec_b[i * W + j + ch * nch];
				temp *= temp;
				norm_i2 += temp;
			}
			norm_loc2 += norm_i2;
		}
		}
		norm2 = max(norm2, norm_loc2/ (kh * kw));
	}	
	}
	return norm2;
}

int isInVector(std::vector<int> v, int value, int from = 0)
{
	for(int i = from; i < (int)v.size(); i++)
	{
		if(v.at(i) == value)
			return i;
	}
	return -1;
		
}


/*int isInVector(std::vector<float> v, float value, int from = 0)
{
	for(int i = from; i < (int)v.size(); i++)
	{
		if(v.at(i) == value)
			return i;
	}
	return -1;
}

int isInVector(std::vector<double> v, double value, int from = 0)
{
	for(int i = from; i < (int)v.size(); i++)
	{
		if(v.at(i) == value)
			return i;
	}
	return -1;
}

int isInVector(std::vector<std::string> v, std::string value, int from = 0)
{
	for(int i = from; i < (int)v.size(); i++)
	{
		if(v.at(i) == value)
			return i;
	}
	return -1;
}*/

template<typename type> int isInVector(std::vector<type> v, type value, int from = 0)
{
	for(int i = from; i < (int)v.size(); i++)
	{
		if(v.at(i) == value)
			return i;
	}
	return -1;	
}

void check_and_add(std::vector<std::string> *v, char *value)
{
	std::string value_string = std::string(value);
	if ( isInVector(*v,value_string, 0) < 0)
		(*v).push_back(value_string);
}

template<typename type> void check_and_add(std::vector<type> *v, type value)
{
	if ( isInVector(*v,value, 0) < 0)
		(*v).push_back(value);
}

void check_and_add(std::vector <std::string> *v, std::vector<std::string> values)
{
	for(int i = 0; i < (int)values.size(); i ++)
	{
		check_and_add(v, values.at(i));
	}
}
void check_and_add(std::vector <std::string> *v, std::vector<std::string> *values)
{
	for(int i = 0; i < (int)values->size(); i ++)
	{
		check_and_add(v, values->at(i));
	}
}
int isInVector(std::vector<std::string> v, const char *value, int from = 0)
{
	for(int i = from; i < (int)v.size(); i++)
	{
		if(v.at(i).compare(value) == 0)
			return i;
	}
	return -1;
}
bool validate_string(std::vector<std::string> list, std::string value)
{
	return isInVector(list, value, 0) >= 0;
}

template<typename type> int maxInVector(std::vector<type> v, int from, int last)
{
	int pos = from;
	for(int i = from; i < (int)v.size() && i<=last; i++)
	{
		if(v.at(i) > v.at(pos))
			pos = i;
	}
	return pos;
}
template<typename type> int maxInVector(std::vector<type> v)
{
	if(v.size() < 1) return -1;
	int pos = 0;
	for(int i = 0; i < (int)v.size(); i++)
	{
		if(v.at(i) > v.at(pos))
			pos = i;
	}
	return pos;
}
		
template<typename type > type sumVector(std::vector<type> v)
{
	type sum = 0;
	for(int i = 0; i < (int)v.size(); i++)
	{
		sum += v.at(i);
	}
	return sum;
}

template<typename type> void xic2ss(std::vector<type> v, double calib_xicxss)
{
	for(unsigned int i = 0; i < v.size(); i++)
	{
		if(v.at(i) > 0)
			v.at(i)  = (type)(calib_xicxss / (double)v.at(i));
	}
}

void printfVector(std::vector<int> v)
{
	for(int i = 0; i < (int) v.size(); i ++)
	{
		printf("%i ", v.at(i));
	}
}
void fprintfVector(FILE *file, std::vector<int> v)
{
	for(int i = 0; i < (int) v.size(); i++)
	{
		fprintf(file, "%i ", v.at(i));
	}
}
void printfVector(std::vector<float> v, const char* format = "%.8f\t")
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		printf(format, v.at(i));
	}
}
void printfVector(std::vector<double> v, const char* format = "%.8f\t")
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		printf(format, v.at(i));
	}
}
void fprintfVector( FILE *file, std::vector<float> v, const char* format = "%.8f\t")
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		fprintf(file, format, v.at(i));
	}
}
void fprintfVector( FILE *file, std::vector<double> v, const char* format = "%.8f\t")
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		fprintf(file, format, v.at(i));
	}
}

void printfVector(std::vector<std::string> v)
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		printf("%s ", v.at(i).data());
	}
}
void fprintfVector( FILE *file, std::vector<std::string> v)
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		fprintf(file, "%s ", v.at(i).data());
	}
}

#endif

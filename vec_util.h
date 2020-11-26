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


int isInVector(std::vector<float> v, float value, int from = 0)
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
		if(v.at(i).compare(value) == 0)
			return i;
	}
	return -1;
}

void check_and_add(std::vector<float> *v, float value)
{
	if ( isInVector(*v, value, 0) < 0)
		(*v).push_back(value);
}
void check_and_add(std::vector<double> *v, double value)
{
	if ( isInVector(*v, value, 0) < 0)
		(*v).push_back(value);
}

void check_and_add(std::vector<std::string> *v, std::string value)
{
	if ( isInVector(*v, value, 0) < 0)
		(*v).push_back(value);
}

void check_and_add(std::vector<std::string> *v, char *value)
{
	std::string value_string = std::string(value);
	if ( isInVector(*v,value_string, 0) < 0)
		(*v).push_back(value_string);
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

template<typename type> void xic2ss(std::vector<type> v, float calib_xicxss)
{
	for(unsigned int i = 0; i < v.size(); i++)
	{
		if(v.at(i) > 0)
			v.at(i)  = calib_xicxss / v.at(i);
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
void printfVector(std::vector<float> v)
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		printf("%.4f ", v.at(i));
	}
}
void printfVector(std::vector<double> v)
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		printf("%.4f ", v.at(i));
	}
}
void fprintfVector( FILE *file, std::vector<float> v)
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		fprintf(file, "%.4f ", v.at(i));
	}
}
void fprintfVector( FILE *file, std::vector<double> v)
{
	for(int i = 0; i < (int)v.size(); i++)
	{
		fprintf(file, "%.4f ", v.at(i));
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

void printTxt(FILE *file, float *f, int H, int W)
{
	for(int i = 0; i < H; i ++)
	{
		for(int j = 0; j < W; j++)
		{
			fprintf(file, "%.16f\t", f[i * W + j]);
		}
		fprintf(file, "\n");
	}
}
void printTxt(FILE *file, double *f, int H, int W)
{
	for(int i = 0; i < H; i ++)
	{
		for(int j = 0; j < W; j++)
		{
			fprintf(file, "%.16f\t", f[i * W + j]);
		}
		fprintf(file, "\n");
	}
}

void printToTxt(const char* file_name, float *f, int H, int W, int nch)
{
	FILE *file = fopen(file_name, "w");
	if (file == NULL)
	{
		dprintf(0, "No valid txt file name\n");
		return;
	}
	for(int ch = 0; ch < nch; ch ++)
	{
		for(int i = 0; i < H; i ++)
		{
			for(int j = 0; j < W; j++)
			{
				fprintf(file, "%.8f\t", f[ch * H * W + i * W + j]);
			}
			fprintf(file, "\n");
		}
	}

}
void printToTxt(const char* file_name, double *f, int H, int W, int nch, const char* format = "%.8f\t")
{
	FILE *file = fopen(file_name, "w");
	if (file == NULL)
	{
		dprintf(0, "No valid txt file name\n");
		return;
	}
	for(int ch = 0; ch < nch; ch ++)
	{
		for(int i = 0; i < H; i ++)
		{
			for(int j = 0; j < W; j++)
			{
				fprintf(file, format, f[ch * H * W + i * W + j]);
			}
			fprintf(file, "\n");
		}
	}
	fclose(file);

}




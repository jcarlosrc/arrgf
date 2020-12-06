#include<algorithm>
#include <stdio.h>
#include <vector>
#include <string>
#include "matrix.h"

namespace Util
{

	template<typename type> int read_input_list(std::vector<type> *values, char **args, int argi, int nargs, bool debug = false, type eps = 0.000001)
	{
		if (debug) printf("Reading list \n");
			
		int go = 1;
		type val0, lastval;
		lastval = -1;
		//printf("i0 = %d added\n", i0);
		std::string link;
		
		while(go == 1)
		{
			val0 = (type)atof(args[argi]);
			argi ++;
			
			if(debug) printf(" val0 = %.2f ", val0);	
			if(val0 != lastval) values -> push_back(val0);
			lastval = val0;
			
			if(!(argi < nargs))
				return argi;
			
			link = std::string(args[argi]);
			argi++;

			if(link == "and")
			{
				go = 1;
			}
			else if (link == ":")
			{
				if (debug) printf(":");
				type d = 1.0;			
				type valend = (type) atof(args[argi]);
				argi ++;
				
				if(argi < nargs)
				{
					// check if there is another : to form a : d : b
					link = std::string(args[argi]);
					argi ++;
					
					if (link == ":" && argi < nargs)
					{
						d = valend;
						valend = atof(args[argi]);
						argi ++;
					}
					else
					{
						argi --;		
					}
				}
				if(debug) printf(" d = %.2f end = %.2f", d, valend);
				// Add value. i0 was already added
				for(type val = val0 + d; val <= valend + eps ; val += d)
				{
					lastval = val;
					values -> push_back(val);
				}
				
				if(argi < nargs)
				{
				
					link = std::string(args[argi]);
					argi++;
					if(link == "and")
					{
						go = 1;
					}
					else
					{
						go = 0;
						argi --;
					}
				}
				else
				{
					return argi;
				}
				
			}
			else
			{
				go = 0;
				argi--;
			}
			
		}
		if(debug) printf("\n");
		return argi;

	}


	int read_input_list(std::vector<int> *values, char **args, int argi, int nargs, bool debug = false)
	{
		if(debug) printf("Reading Integer list \n");
			
		int go = 1;
		int val0, lastval;
		lastval = -1;
		//printf("i0 = %d added\n", i0);
		std::string link;
		
		while(go == 1)
		{
			val0 = atoi(args[argi]);
			argi ++;
			
			if(val0 != lastval) values -> push_back(val0);
			lastval = val0;
			
			if(debug) printf("val %d added\n", val0);
			
			if(!(argi < nargs))
				return argi;
				
			link = std::string(args[argi]);
			argi++;

			if(link == "and")
			{
				go = 1;
			}
			else if (link == ":")
			{
				printf(":");
				int d = 1;			
				int valend = atoi(args[argi]);
				argi ++;
				
				if(argi < nargs)
				{
					// check if there is another : to form a : d : b
					link = std::string(args[argi]);
					argi ++;
					
					if (link == ":" && argi < nargs)
					{
						d = valend;
						valend = atoi(args[argi]);
						argi ++;
					}
					else
					{
						argi --;		
					}
				}
				// Add value. i0 was already added
				for(int val = val0 + d; val < valend+1 ; val += d)
				{
					lastval = val;
					values -> push_back(val);
					if(debug) printf("val %d added\n", val);
				}
				
				if(argi < nargs)
				{			
					link = std::string(args[argi]);
					argi++;
					if(link == "and")
					{
						go = 1;
					}
					else
					{
						go = 0;
						argi --;
					}
				}
				else
				{
					return argi;
				}
				
			}
			else
			{
				go = 0;
				argi--;
			}
			
		}
		if(debug) printf("\n");
		return argi;

	}

	/* Read string list only in format xx and xxx and xxxx */
	int read_input_list(std::vector<std::string> *values, char **args, int argi, int nargs, bool debug = false)
	{
		if(debug) printf("Reading Integer list \n");
			
		int go = 1;
		std::string val0, lastval;
		lastval = "";
		//printf("i0 = %d added\n", i0);
		std::string link;
		
		while(go == 1)
		{
			val0 = std::string(args[argi]);
			argi ++;
			
			if(val0 != lastval) values -> push_back(val0);
			lastval = val0;
			
			if(debug) printf("val %s added\n", val0.data());
			
			if(!(argi < nargs))
				return argi;
				
			link = std::string(args[argi]);
			argi++;

			if(link == "and" || link == ",")
			{
				go = 1;
			}
			else
			{
				go = 0;
				argi--;
			}
			
		}
		if(debug) printf("\n");
		return argi;

	}

	int is_in(std::string value, std::vector<std::string> values)
	{
		for(unsigned int i = 0; i < values.size(); i++)
		{
			if (value == values.at(i))
			{
				return (int) i;
			}
		}
		return -1;
	}

	/* Read string list only in format xx and xxx and xxxx */
	int read_and_validate_input_list(std::vector<std::string> *values, std::vector<std::string> *white_list, char **args, int argi, int nargs, bool debug = false)
	{
		if(debug) printf("Reading Integer list \n");
			
		int go = 1;
		std::string val0, lastval;
		lastval = "";
		//printf("i0 = %d added\n", i0);
		std::string link;
		
		while(go == 1)
		{
			val0 = std::string(args[argi]);
			argi ++;
			
			if(val0 != lastval && is_in(val0, *white_list) < 0) values -> push_back(val0);
			lastval = val0;
			
			if(debug) printf("val %s added\n", val0.data());
			
			if(!(argi < nargs))
				return argi;
				
			link = std::string(args[argi]);
			argi++;

			if(link == "and" || link == ",")
			{
				go = 1;
			}
			else
			{
				go = 0;
				argi--;
			}
			
		}
		if(debug) printf("\n");
		return argi;

	}

	template<typename type>	void printToTxt(const std::string file_name, type *f, int H, int W, int nch = 1, const char* format = "%.8f\t", int rgb_type = MATRIX_RGB_TYPE_CHUNK)
	{
		// Print each chanel of image in a txt file
		for(int ch = 0; ch < nch; ch++)
		{
			std::string file_ch_name = file_name + "_" + Patch::to_string(ch) + ".txt" ;
			FILE *file = fopen( file_ch_name.data() , "w");
			if (file == NULL)
			{
				dprintf(0, "No valid txt file name\n");
				return;
			}

			for(int i = 0; i < H; i ++)
			{
				for(int j = 0; j < W; j++)
				{
					switch(rgb_type){
						case MATRIX_RGB_TYPE_CHUNK:
						fprintf(file, format, f[ch * H * W + i * W + j]);
						break;

						case MATRIX_RGB_TYPE_INTER:
						fprintf(file, format, f[nch * i * W + nch * j + ch]);
						break;
					}
				}
				fprintf(file, "\n");
			}
			dprintf(0, "\n\t > %s", file_ch_name.data());
			fclose(file);
		}
	}

}




//Cuda implementation of the RRGF filter
#include<algorithm>
#include <stdio.h>
#include <math.h>
#include <chrono>

#include "png_tool.hpp"
#include <string>
#include <vector>
#include <ctime>

#include "convolution_kernel.h"
//#include "bilateral_kernel.h"
//#include "stddev_kernel.h"
#include "common_math_kernel.h"
#include "patch.hpp"
#include "local_measure.h"
#include "trilateral_kernel.h"
#include "rgbnorm2_kernel.h"
#include "l2loc2_kernel.h"
#include "vec_util.h"
#include "util.h"

// Include color space conversion
//#include "ColorSpace-master/src/ColorSpace.h"
//#include "ColorSpace-master/src/Comparison.h"
//#include "ColorSpace-master/src/Conversion.h"
//#include "ColorSpace-master/src/Conversion.h"


//Data type to use for images and images in kernels
#define type float

//Block dimensions for Bil
const int blockBilH = 4;
const int blockBilW = 16;


//Size of blocks for convolution
const int blockConvH = 16;
const int blockConvW = 16;


// Generic Block Sizes
int gbsX = 16;
int gbsY = 16;
int gbsZ = 1;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#ifndef CTE255
#define CTE255 255.0f
#endif


int debug = 0;
std::string debug_name = std::string("no");

std::vector<int> it_list;	
//showHelp();

std::string input_name = std::string("");
std::string output_name = std::string("");


std::string it_from_image_name = std::string("");
int it_from = 0;
bool load_it = false;
bool load_it_gammacor = true;

int make_output_default = 1;

// Default parameters

std::string commands_string = "";

// Theoretical frequency cutoff for avoiding aliasing
type cutoff = 0.5;
type xic = 1.0/8;
type calibration_xicxss = 0.4375;
bool xic_exp = false;


std::vector<std::string> conf_all;
std::vector<std::string> conf_list;

std::vector<type> scaleValues;
int scale_exp = 0;
//type scale = 1.0f;
int scale_back = false;
std::string scale_back_name = std::string("no");

// Spatial kernel specs
std::vector<type> ssValues;
int ss_exp = 0;
int ss_mod = 1;

int sker_mod = 1;	// allow modifications?
int sker_exp = 0;	// is explicitly defined?
std::string sker_name = std::string("gaussian");
int sker = 0; // gaussian exp(-0.5x**2/ss**2) spatial kernel by default
//type ss =  0.355 / cutoff;	// sigma value for gaussian
int sskh = 0;
int sskw = 0;

std::vector<type> srValues;
int sr_exp = 0;
int sr_mod = 1;

int rker_mod = 1;
int rker_exp = 0;
std::string rker_name = std::string("gaussian");
int rker = 0; // gaussian intensity kernel by default	

int regker_mod = 1;
int regker_exp = 0;
std::string regker_name = std::string("gaussian");
int regker = 0; // gaussian regularization kernel by default
type sreg = 0.25 ;
std::vector<type> sregValues;
int sreg_mod = 1;
int sreg_exp = 0;
int regkh = 0;
int regkw = 0;

int nit = 10;
int nit_mod = 1;
int nit_exp = 0;

bool print_gamma = false;
bool gammacor_in = true;
bool print_linear = false;
bool gammacor_load_it = true;
bool gammacor_txt_out = false;	// Txt is saved in linear space. ALWAYS.

int adaptive_mod = 1;
std::string adaptive_name = "yes";
int adaptive_exp = 0;
int adaptive = 1;

// Local measure for adaptativeness
int local_measure = 0; // max
std::string local_measure_name = std::string("max");
type sr_ref = 0.48;
int local_measure_mod = 1;
int local_measure_exp = 0;

// Local measure weights 
std::string lweights_name = std::string("gaussian");
int lweights = 2;	// Gaussian, to achieve smoother transitions. Specially important for lm = ma
int lweights_mod = 1;
bool lweights_exp = false;
int lkh = 0;
int lkw = 0;

type sl;
int sl_mod = 1;
int sl_exp = 0;

// Regularization convolution for local measure (like gaussian blur to smooth results)
std::string lmreg_ker_name = std::string("none");
int lmreg_ker = -1;
type lmreg_s = 0.0f;
bool lmreg_s_exp = false;
int lmreg_ker_mod = 1;
int lmreg_ker_exp = 0;
int lmreg_kh;
int lmreg_kw;



int mker_mod = 1;
int mker_exp = 0;
std::string mker_name = std::string("none");
int mker = -1; // No median kernel by default
type sm = 0;
int sm_mod = 1;

int conf_mod = 1;
int conf_exp = 0;
int conf = 0; // arrgf by default
std::string conf_name = std::string("arrgf");

type M = 20;

type infsr = 0.0001;
bool infsr_exp = false;

int i0 = 10;
int iend = 10;
int print_exp = 0;
int max_it = -1;
bool print_all = false;

// conv_norm_list < calc_conv_norm_list < check_conv_norm_list < show_norm_list
std::vector <std::string> conv_norm_list ;	// Norms for which converging means the algorithm converged and we do not check convergence anymore. All have to converge to say this.
std::vector  <std::string> calc_conv_list;	// Norms for which we require the algorithm until they converge. This is they make calc_norms = true. = conv_norm_list by default.
std::vector  <std::string> check_conv_norm_list;	// Norms for we check convergence, they not influence the lifetime of the algorithm. = show_norm_list by default
std::vector <std::string> show_norm_list ;	// Norms for which we calculate values at each iteration to show in screen.

std::vector < int> show_conv_values ;	// Saves the position of epsilon to check convergence for show_norm_list
std::vector < int> conv_values;			// Saves the position of epsilon to check convergence for conv_norm_list;

std::vector < type> conv_eps_list;	// vector for epsilon values for convergence;
bool conv_eps_exp = false;	//true if we have set manually
type conv_eps_default = 0.01;	// Default value for convergence. if all norms in conv_norm_list < conv_eps then we have converged

std::string conv_norm_default = std::string("l2loc");	// Default norm in conv_norm_list;

// All norms available
std::vector<std::string> all_norms;
bool conv_norm_exp = false;

std::vector<type> conv_eps_vector;


bool calc_norms = false;
bool force_stop_on_convergence = false;
bool stop_showing_on_convergence = false;

bool show_norm_exp = false;
bool show_norm_all = false;

type show_conv_eps_default = 0.01;
bool show_conv_eps_exp = false;

bool show_conv = false;
bool print_alt_conv = false;
bool calc_convergence = false;

std::vector<std::string> conf_vector;

int show_conv_max = 0;
int show_conv_l2loc = 0;
int show_conv_l2 = 0;

int conv_l2 = 0;
int conv_l2loc = 0;
int conv_max = 0;
// Save log to file also:
bool save_log = false;
FILE *log_file;
std::string log_file_name;
bool auto_log_string = true;


std::string help_string;

std::vector<int> h_slice_list;
std::vector<int> v_slice_list;

bool save_txt = false;
// Print |g - f|(r, g, b) differences image -> (r, g, b) image
bool print_diff_gamma = false;
bool print_diff_linear = false;
// Print |g - f|_{rgb} differences -> single_channel image
bool print_diff_single_linear = false;
bool print_diff_single_gamma = false;

bool txt_local_measure = false;
bool print_local_measure_gamma = false;
bool print_local_measure_linear = false;


// *** Miscelaneus algorithms for tesis
bool make_contrast_enhacement = false;
bool make_contrast_enhacement_linear = false;
std::vector<type> ce_factor_list;

// ** Conversion to lab for better calculations
//bool lab = false;

// ** Adapt sr value for RGF
bool adapt_sr = true;

// *******************************************************************
// auxiliary functions
// ********************************************************************

int showHelp()
{
	printf("%s\n", help_string.data());
	printf("\nBASIC CALL:");
	printf("\n\n\t./arrgf.out -input <input> -<conf = arrgf> -ss <ss = 3.5> -sr <sr = 0.1> -sreg <sreg = 0.25> -print-it <iterations = 10>");
	printf("\n\n* DO NOT append the extension '.png' to the input file name.");
	printf("\n* Input is mandatory.");

	printf("\n\n EXAMPLE:\n");
	printf("\n\t ./arrgf.out -input /home/test-image -arrgf -ss 3.0 and 3.5 -sr 0.1 and 0.2 -sreg 0.25 -print-it 1 : 2 : 5 and 8");
	printf("\n\n This calculates ARRGF filter for spatial Gaussian(ss = 3.0 and 3.5),  range Gaussian(sr = 0.1 and 0.2), regularization Gaussian(0.25 x ss = 3.0/4) and will print iterations 1, 1+2 = 3, 1+2+2 = 5 and 8");
	printf("\n Output file is in format <input file>-<conf>-sr-<sr>-ss-<ss>-it-<it>.png.");
	printf("\n\nOPTIONS");
	printf("\n*-output <output> will specify output files in the format output-<conf>-sr-<sr>-ss-<ss>-it-<it>.png");
	printf("\n* -txt :  We save to TXT in linear space.");
	printf("\n* -log : We save a .txt log file. This will save a output-LOG-<parameters>.txt file");
	printf("\n -<conf>: rgf, bilateral, rrgf, conv and arrgf. Convolution is only currently available for Gaussian and box kernels (ss parameter). Using a configuration option will disable some parameters to achieve such filter. Example: \n\n ./arrgf-rgb -input /home/image.png -rgf -print-it 5 -ss 3.0 -sr 0.1\n\nwill get RGF filter, print iteration 5 for spatial Gaussian(3.0) and range Gaussian(0.1)");
	printf("\n\n INPUT / OUTPUT GAMMA: -linear-input, -linear-output, -linear-output-also to read/write in linear space. Gamma correction is enabled for input/output by default.");
	printf("\n\t Gamma correction for input / output is enabled by default.");
	printf("\n\n KERNELS:");
	printf("\n\t -sker : kernel for spatial calculation, default ss-Gaussian(ss = 3.5)");
	printf("\n\t -ss : sigma for spatial kernel, default 3.5");
	printf("\n\t -rker : kernel for range: sr-Gaussian(sr = 0.1)");
	printf("\n\t -sr : sigma for range kernel, default sr = 0.1. sr = 0 means only spatial convolution.");
	printf("\n\t -regker :  Regularization kernel, default s-Gaussian(s = sreg * ss, sreg = 0.25)");
	printf("\n\t -sreg : Regularization s / ss, default 0.25. sreg = 0 means no regularization.");
	printf("\n\t -lm :  local measure, default max");
	printf("\n\t -lw : local weights for local measure, default ss-Gaussian");
	
	printf("\n\n OTHER OPTIONS:");
	printf("\n* We also have median kernel: mker, ms. Please refer to code or rcjcarlos@gmail.com for more information.");

	printf("\n\n CONVERGENCE: \n\t* We can make the algorithm to run until convergence specifying some convergence norms (-conv-norm) and/or some convergence eps values (-conv-eps). The algorithm will run until convergence is achieved for ALL those norms / eps values. Error values will be showed for each iteration and iterations achieving convergence will be printed. This does *NOT* prevent the algorithm to reach the iterations set with -print <iterations>. We can force to stop on convergence by using -force-stop-on-convergence or -force-stop.\n\t* We can specify a maximum iteration with max-it <max it> to stop the algorithm independently of convergence. It will also prevent other print <value> higher than such max-it value.");
	printf("\nOPTIONS: Currently three stop-norm options are accepted:\n\t*l2loc : maximum of l2 norms over neighboorhoods of every pixel. Such regions are the same as defined by the Spatial kernel.\n\t*l2, which takes the l2 norm\n\t*max which takes the maximum value over every pixel.\n");
	printf("\n* We can show error values , check convergence and print convergence iterations for other norm without forcing to achieve convergence with -show-norm <norms> / -show-conv-norm <norms> / -show-conv-print <norms>. This will print errors and check convergence but wont force the algorithm to achieve it.");
	printf("\nEXAMPLE:\n\n\t/arrgf.out -input ../barbara.png -ss 2.84 -sr 0.5 -conv-norm max -conv-eps 0.001 -show-norm l2 -print-conv\n\nThis will run the algorithm until convergence is achieved for max (and print convergence) but also print iterations which converge for l2\n" );
	printf("\nDEFAULT CONVERGENCE VALUES: Default values for -stop-norm is l2loc, for -stop-eps is 0.001 if some convergence command is used.");
	
	printf("\n\n OTHER FEATURES:");
	printf("\n\t -print-diff-gamma, -print-diff-linear print differences with input function in gamma and linear color spaces.");
	printf("\n\t -print-diff-single-gamma, -print-diff-single-linear to print single-channel differences which are calculated using RGB/sqrt(n) norm, we use .");
	printf("\n\t-slice-h / -slice-v <int values> to print slices (a row or column of results). Slices are saved to .txt files in linear space.");
	printf("\n\n");
	return 1;
}

void show(type *gpuArray, int n, std::string message)
{
	
	type *hostArray = new type[n];
	cudaMemcpy(hostArray, gpuArray, n*sizeof(type), cudaMemcpyDeviceToHost);
	printf("%s", message.data());
	printf("Memory direction: %li\n", (long int)&gpuArray);
	for(int i = 0; i < n; i++)
		printf("%.2f\t", hostArray[i]);
	printf("\n");
	
	delete[] hostArray;
}

int not_zeros(type *devx, int n, std::string message = std::string(""))
{
	type *hostx = new type[n];
	cudaMemcpy(hostx, devx, n*sizeof(type), cudaMemcpyDeviceToHost);
	printf("%s", message.data());
	
	int cont = 0;
	for(int i = 0; i < n; i++)
	{
		if(hostx[i] != 0) cont++;
	}
	return cont;
}

int equal_zeros(type *devx, int n, std::string message = std::string(""))
{
	type *hostx = new type[n];
	cudaMemcpy(hostx, devx, n*sizeof(type), cudaMemcpyDeviceToHost);
	printf("%s", message.data());
	
	int cont = 0;
	for(int i = 0; i < n; i++)
	{
		if(hostx[i] == 0) cont++;
	}
	return cont;
}

// Set configuration from configuration name: rgf, rrgf, arrgf, conv
int set_conf ( std::string conf_string )
{
	if(conf_string.compare(std::string("bilateral")) == 0)
	{
		conf = 3;
		adaptive = 0; adaptive_mod = 0;
		regker = -1; sreg = 0; regker_name = std::string("none"); regker_mod = 0;
		mker = -1; sm = 0; mker_name = std::string("none"); mker_mod = 0;
		local_measure = -1; sl = 0; local_measure_name = "none"; local_measure_mod = 0; sl_mod = 0;
		if(nit_mod && !nit_exp) nit = 1;
		
		conf_mod = 0;
		conf_exp = 1;
		
		conf_name = conf_string;
		return conf;
	}
	if(conf_string.compare(std::string("rgf")) == 0)
	{
		conf = 2;
		adaptive = 0; adaptive_mod = 0;
		regker = -1; sreg = 0; regker_name = std::string("none"); regker_mod = 0;
		mker = -1; sm = 0; mker_name = std::string("none"); mker_mod = 0;
		local_measure = -1; sl = 0; local_measure_name = "none"; local_measure_mod = 0; sl_mod = 0;
		if(nit_mod && !nit_exp) nit = 10;
		
		conf_mod = 0;
		conf_exp = 1;
		conf_name = conf_string;
		return conf;
	}
	if(conf_string.compare(std::string("rrgf")) == 0)
	{
		conf = 1;
		adaptive = 0; adaptive_mod = 0;
		mker = -1; sm = 0; mker_name = std::string("none"); mker_mod = 0;
		local_measure = -1; sl = 0; local_measure_name = "none"; local_measure_mod = 0;
		if(nit_mod && !nit_exp) nit = 10;
		
		conf_mod = 0;
		conf_exp = 1;
		conf_name = conf_string;
		return conf;
	}
	if(conf_string.compare(std::string("arrgf")) == 0 || conf_string.compare(std::string("argf")) == 0)
	{
		conf = 0;
		adaptive = 1; adaptive_mod = 0;
		mker = -1; sm = 0; mker_name = std::string("none"); mker_mod = 0;
		if(nit_mod && !nit_exp) nit = 10;
		
		conf_mod = 0;
		conf_exp = 1;
		conf_name = conf_string;
		return conf;
	}
	if(conf_string.compare(std::string("conv")) == 0 || conf_string.compare(std::string("convolution")) == 0)
	{
		conf = 4;
		adaptive = 0; adaptive_mod = 0;
		regker = -1; sreg = 0; regker_name = std::string("none"); regker_mod = 0;
		mker = -1; sm = 0; mker_name = std::string("none"); mker_mod = 0;
		local_measure = -1; sl = 0; local_measure_name = "none"; local_measure_mod = 0;
		rker = -1; rker_name = "none"; rker_mod = 0;
		if(nit_mod && !nit_exp) nit = 1;
		
		conf_mod = 0;
		conf_exp = 1;
		conf_name = conf_string;
		return conf;
	}
	return conf;
}
/*void save_to_grayimage(std::string pngname, type *devx, int H, int W)
{
	unsigned int color_type = 1;
	unsigned int bit_depth = 8;
	type* hostx = new type[H * W];
	cudaMemcpy(hostx, devx, H*W*sizeof(type), cudaMemcpyDeviceToHost);	
	
	//printf("Copy new data to pixels buffer applying back gamma correction... ");
	unsigned char* pixels = new unsigned char [H * W];
	unsigned char* p = pixels;
	for(int i = 0; i< H*W; i++){
		*p = (unsigned char)(255.0f * hostx[i]);
		p++;
	}
	save_image(pngname.data(), pixels, H, W, 1, color_type, bit_depth);
	delete[] pixels;
	delete[] hostx;
	//printf("done.\n");

}*/

//*******************************************************************************
// Main
// ******************************************************************************


int main(int nargs, char** args){	

	// Initialize available norms
	all_norms.push_back("l2");
	all_norms.push_back("l2loc");
	all_norms.push_back("max");
	
	// Initialize valid algorithms
	conf_all.push_back("arrgf");
	conf_all.push_back("rrgf");
	conf_all.push_back("rgf");
	conf_all.push_back("bilateral");
	conf_all.push_back("conv");

	// Make commands_string to inform log_file
	
	for(int i = 0; i < nargs; i ++)
	{
		commands_string += args[i];
		if (i + 1 < nargs)
			commands_string += " ";
	}
	if(debug) printf("Commands entered: %s\n", commands_string.data());
	// Check parameters
	int argi = 1;
	while(argi < nargs)
	{	
	
		std::string pname = std::string(args[argi]);
		argi ++;
		
		//if(argi >= nargs) break;

		if(pname == "-adapt-sr" || pname == "-asr")
		{
			adapt_sr = true;
		}
	
		if(pname.compare("-load-it") == 0)
		{
			if(argi + 3 < nargs)
			{
				load_it = true;
				it_from = atoi(args[argi]);
				argi ++;
				
				std::string next_arg = std::string(args[argi]);
				argi ++;
				if(next_arg.compare("from-png") == 0 || next_arg.compare("from-image") == 0 || next_arg.compare("from-im") == 0)
				{
					it_from_image_name = std::string(args[argi]);
					argi++;
				}
				if(next_arg.compare("from-png-linear") == 0 || next_arg.compare("from-image-linear") == 0 || next_arg.compare("from-im-lin") == 0)
				{
					gammacor_load_it = false;
					it_from_image_name = std::string(args[argi]);
					argi++;
				}
			}

		}
		
		if(pname == "-print-diff" || pname == "print-diff-gamma")
		{
			print_diff_gamma = true;
		}
		if(pname == "-print-diff-only" || pname == "print-diff-gamma-only")
		{
			print_diff_gamma = true;
			print_gamma = false;
			print_linear = false;
		}
		if(pname == "-print-diff-linear")
		{
			print_diff_linear = true;
		}
		if(pname == "-print-diff-linear-only")
		{
			print_diff_linear = true;
			print_gamma = false;
			print_linear = false;
		}
		
		if(pname == "-hslice" || pname == "-h-slice"|| pname == "-vslice" || pname == "-v-slice" || pname == "-slice-v" || pname == "-slice-h")
		{
			std::vector<int> *slice_list;
			
			if(pname == "-hslice" || pname == "-h-slice" || pname == "-slice-h" || pname == "-print-slice-h" )
			{
				slice_list = &h_slice_list;
			}
			if( pname == "-vslice" || pname == "-v-slice" || pname == "-slice-v" || pname == "-print-slice-v")
			{
				slice_list = &v_slice_list;
			}	
			printf(" Reading slice list : \n");
			
			slice_list->clear();
			printf("List cleared\n");
			argi = read_input_list(slice_list, args, argi, nargs);

		}
		if(pname == "-slice-only" || pname == "-only-slices" || pname == "print-slice-only")
		{
			print_gamma = false;
			print_linear = false;
		}

		if(pname.compare("-txt") == 0 || pname.compare("-save-txt") == 0)
		{
			save_txt = true;
		}
		if(pname == "-txt-only" || pname == "-save-txt-only")
		{
			save_txt = true;
			print_gamma = false;
			print_linear = false;
		}
		if (pname.compare("-test") == 0)
		{
			printf("-test\n");
			input_name = "../images/barbara/barbara.png";
			output_name = "../test/barbara";
			
			std::string conf_string = std::string(args[argi]);
			argi++;

			set_conf(conf_string);
			// Deafult parameters values
			ssValues.push_back(3.0);
			if (conf == 0) srValues.push_back(0.5);
			else if (conf == 1 || conf ==2) srValues.push_back(0.05);
			scaleValues.push_back(1.0);
			
		}
		
		if(pname.compare("-conf") == 0)
		{
			if(pname.compare("all") == 0)
			{
				check_and_add(&conf_list, conf_all);
			}
			else
			{
				if(conf_mod)
				{				
					std::string conf_string = std::string(args[argi]);
					argi ++;

					set_conf(conf_string);
				}
			}
			printf("conf\n");

		}
		// Express configurations
		if(pname.compare("-arrgf") == 0)
		{
			if(conf_mod)
			{
				set_conf(std::string("arrgf"));
			}
		}
		if(pname.compare("-rrgf") == 0)
		{
			if(conf_mod)
			{
				set_conf(std::string("rrgf"));
			}
		}
		if(pname.compare("-rgf") == 0)
		{
			if(conf_mod)
			{
				set_conf(std::string("rgf"));
			}
		}
		if(pname.compare("-conv") == 0 || pname.compare("-convolution") == 0)
		{
			if(conf_mod)
			{
				set_conf(std::string("conv"));
			}
		}
		if(pname.compare("-bilateral") == 0|| pname.compare("-bil") == 0)
		{
			if(conf_mod)
			{
				set_conf(std::string("bilateral"));
			}
		}
		// Stack for showing norm values between iterations. 
		if(pname.compare("-show-norm") == 0 || pname.compare("-show-norms") == 0 || pname.compare("-show-conv-print-norm") == 0 || pname.compare("-show-conv-print-norms") == 0 || pname.compare("-show-conv-print") == 0|| pname.compare("-show-conv-norms") == 0 || pname.compare("-show-conv") == 0 || pname.compare("-show-conv-norm") == 0)
		{
			
			show_norm_exp = true;
			if(pname.compare("-show-conv-norms") == 0 || pname.compare("-show-conv") == 0 || pname.compare("-show-conv-norm") == 0)
			{
				show_conv = true;
			}
			if(pname.compare("-show-conv-print") == 0 || pname.compare("-show-conv-print-norms") || pname.compare("-show-conv-print-norm") == 0)
			{
				show_conv = true;
				print_alt_conv = true;
			}
			
			std::string norm_string = std::string(args[argi]);
			argi ++;
			
			if (norm_string.compare("all") == 0)
			{
				//norm_string = std::string("l2");
				check_and_add(&show_norm_list, all_norms);
			
			}
			else if( validate_string(all_norms, norm_string) )
			{
				check_and_add(&show_norm_list, norm_string);
					
				int go = 1;

				
				while(go == 1 && argi < nargs)
				{
					std::string temp = std::string(args[argi]);
					argi++;

					if(temp.compare("and") == 0 || temp.compare(",") == 0 && argi+1 < nargs)
					{
						norm_string = std::string(args[argi]);
						argi++;
						check_and_add(&show_norm_list, norm_string);
						/*if(isInVector(norm_vector, norm_string, 0 ) < 0 )
						{
							printf("\nAdding %s norm to stack", norm_string.data());
							norm_vector.push_back(norm_string);
						}*/
							
						go = 1;
					}
					else
					{
						argi--;
						go = 0;
					}			
				}	
			} 
			else
			{
				argi --;
			}
		}
		// add ALL norms for showing
		if (pname.compare("-show-norm-all") == 0 || pname.compare("-show-all-norms") == 0 || pname.compare("-show-conv-print-all") == 0 || pname.compare("-show-conv-all") == 0)
		{			
			calc_norms = true;
			check_and_add(&show_norm_list, all_norms);
			
			if(pname.compare("-show-conv-all") == 0)
			{
				show_conv = true;
			}
			if(pname.compare("-show-conv-print-all") == 0)
			{
				show_conv = true;
				print_alt_conv = true;
			}
			
			show_norm_exp = true;

		}
		
		if(pname.compare("-show-conv") == 0)
		{
			show_conv = true;
		}
		
		if(pname.compare("-show-conv-print") == 0)
		{
			show_conv = true;
			print_alt_conv = true;
		}

		
		if(pname.compare("-conv-norm") == 0 || pname.compare("-conv-norms") == 0 || pname.compare("-convergence-norms") == 0 || pname.compare("-convergence-norm") == 0)
		{
			
			conv_norm_exp = true;
			calc_norms = true;
			
			std::string norm_string = std::string(args[argi]);
			argi ++;
			
			if (norm_string.compare("all") == 0)
			{
				check_and_add(&conv_norm_list, all_norms);
			
			}
			else if( validate_string(all_norms, norm_string) )
			{
				check_and_add(&conv_norm_list, norm_string);
					
				int go = 1;

				
				while(go == 1 && argi < nargs)
				{
					std::string temp = std::string(args[argi]);
					argi++;

					if(temp.compare("and") == 0 || temp.compare(",") == 0 && argi+1 < nargs)
					{
						norm_string = std::string(args[argi]);
						argi++;
						check_and_add(&conv_norm_list, norm_string);
							
						go = 1;
					}
					else
					{
						argi--;
						go = 0;
					}			
				}	
			} 
			else
			{
				argi --;
			}
		}
		
		if(pname.compare("-conv-eps") == 0)
		{
			
			conv_eps_exp = true;
			calc_norms = true;
			
			type eps_value = atof(args[argi]);
			argi ++;
			
			check_and_add(&conv_eps_list, eps_value);
				
			int go = 1;

			
			while(go == 1 && argi < nargs)
			{
				std::string temp = std::string(args[argi]);
				argi++;

				if(temp.compare("and") == 0 || temp.compare(",") == 0 && argi+1 < nargs)
				{
					type eps_value = atof(args[argi]);
					argi++;
					check_and_add(&conv_eps_list, eps_value);
						
					go = 1;
				}
				else
				{
					argi--;
					go = 0;
				}			
			}	
		}

		// add ALL norms to the convergence criteria
		if (pname.compare("-conv-norm-all") == 0 || pname.compare("-conv-all-norms") == 0)
		{			
			
			conv_norm_exp = true;
			check_and_add(&conv_norm_list, all_norms);
			calc_norms = true;

		}

		if (pname.compare("-log") == 0 || pname.compare(">") == 0)
		{
			auto_log_string = false;
			save_log = true;
			if(argi < nargs)
			{
				log_file_name = std::string(args[argi]);
				argi ++;
			}
			
			if (log_file_name.compare("auto") == 0)
			{
				auto_log_string = true;
				log_file_name = "";
			}
		}
		
		if(pname.compare("-log-auto") == 0 )
		{
			auto_log_string = true;
			save_log = true;
		}
		
		if (pname.compare("-log-auto") == 0 || pname.compare("-make-log") == 0)
		{
			save_log = true;
			log_file_name = std::string("");
		}
		
		if( pname.compare("-no-stop-showing-on-convergence") == 0 || pname.compare("-no-stop-showing-on-conv") == 0)
		{
			stop_showing_on_convergence = false;
		}
		if( pname.compare("-stop-showing-on-convergence") == 0 || pname.compare("-stop-showing-on-conv") == 0)
		{
			stop_showing_on_convergence = true;
		}
		
		if( pname.compare("-stop-on-conv") == 0 || pname.compare("-stop-on-convergence") == 0 || pname.compare("-stop-conv") == 0 || pname.compare("-force-stop-on-convergence") == 0 || pname.compare("-fs") == 0 || pname.compare("-force-stop") == 0 || pname.compare("-force-stop-on-conv") == 0)
		{
			calc_norms = true;
			force_stop_on_convergence = true;
		}
		
		if(pname.compare("-max-it") == 0)
		{
			max_it = atoi(args[argi]);
			printf("-max-it");
			argi++;
		}
		// Input file
		if(pname.compare(std::string("-input")) == 0 || pname.compare(std::string("-in")) == 0)
		{
			//printf("Input : ");
			input_name = std::string(args[argi]);
			argi++;
			//printf("%s\n", input_name.data());
		}
		// Output file
		if(pname.compare(std::string("-output")) == 0 || pname.compare(std::string("-out")) == 0)
		{
			//printf("Output: ");
			output_name = std::string(args[argi]);
			argi++;
			//printf("%s\n", output_name.data());
			make_output_default = 0;
			
		}
		// Check for iterations to print in the format x1 and x2 and x3 to x4 to x5 and x6 ... 
		if(pname.compare("-print-it") == 0 || pname.compare("-print-it-linear") == 0 || pname.compare("-print-it-gamma") == 0 || pname == "-nit" || pname == "-it" || pname == "-calc-it" /* Legacy */)
		{
			if(pname.compare("-print-it-linear") == 0 || pname.compare("-print-linear") == 0)
			{
				print_linear = true;
			}
			if(pname.compare("-print-it-gamma") == 0 || pname.compare("-print-it-gamma") == 0 || pname.compare("-print-it") == 0 )
			{
				print_gamma = true;
			}
			std::string option = std::string(args[argi]);
			if(option.compare("all") == 0)
			{
				print_all = true;
				argi++;
			}
			else
			{
				print_exp = 1;
				it_list.clear();
				argi = read_input_list(&it_list, args, argi, nargs);
			}

		}
		if(pname == "-print-its-gamma" || pname == "-print-its" || pname == "-print-iterations-gamma")
		{
			print_gamma = true;
		}
		if(pname == "-print-its-linear" || pname == "-print-iterations-linear")
		{
			print_linear = true;
		}
		if(pname.compare("-print-all") == 0 || pname.compare("-print-all-gamma") == 0)
		{
			print_all = true;
			print_gamma = true;
		}
		if(pname.compare("-print-all-linear") == 0)
		{
			print_all = true;
			print_linear = true;
			
		}
		
		// Check for list of sr values
		if(pname.compare("-sr") == 0)
		{
			srValues.clear();
			argi = read_input_list(&srValues, args, argi, nargs);
			sr_exp = true;
		}
		
		// Check for list of ss values
		if(pname.compare("-ss") == 0)
		{
			ssValues.clear();
			argi = read_input_list(&ssValues, args, argi, nargs);			
		}
		// Regularization
		if(pname.compare(std::string("-sreg")) == 0 || pname.compare("-s") == 0 || pname.compare("-reg-sigma") == 0 || pname.compare("-regularization-sigma") == 0 || pname.compare("-regs") == 0)
		{
			if(sreg_mod )
			{
				sreg_exp = true;
				sregValues.clear();
				argi = read_input_list(&sregValues, args, argi, nargs);
			}
		}
		if (pname == "-xic")
		{
			ssValues.clear();
			argi = read_input_list(&ssValues, args, argi, nargs);
			xic2ss<type>(ssValues, calibration_xicxss);
			xic_exp = true;
		}
		
		// Change xicxss
		if (pname == "-xicxss" || pname == "-calibration-xicxss")
		{
			calibration_xicxss = atof(args[argi]);
			argi++;
		}

		// Check for list of scale values

		if(pname.compare("-scale") == 0)
		{
			scaleValues.clear();
			argi = read_input_list(&scaleValues, args, argi, nargs);
			scale_exp = true;

		}
		
		if (pname.compare("-sb") == 0 || pname.compare("-scale-back") == 0)
		{
			scale_back = 1;
			scale_back_name = "yes";
		}
		
		// Show help
		if(pname.compare(std::string("-help")) == 0)
		{
			showHelp();	
			argi++;
		}
		
		if(pname.compare(std::string("-infsr")) == 0)
		{
			infsr_exp = true;
			infsr = atof(args[argi]);
			argi++;
		}
		/*if(pname.compare(std::string("scale")) == 0)
		{
			scale = atof(args[argi]);
			argi++;
		}*/
		if(pname.compare(std::string("-cutoff")) == 0)
		{
			cutoff = atof(args[argi]);
			argi++;
		}
		if(pname.compare(std::string("-lm")) == 0 || pname.compare(std::string("-local-measure")) == 0 || pname == "-local_measure" || pname == "-lmker" || pname == "local-measure-kernel")
		{
			std::string temp = std::string(args[argi]);
			argi ++;

			if(temp.compare(std::string("max")) == 0)
			{
				local_measure = 0;
				local_measure_name = temp;
				local_measure_exp = 1;
				sr_ref = 0.48;
			}
			if(temp.compare(std::string("mean")) == 0)
			{
				local_measure = 1;
				local_measure_name = temp;
				local_measure_exp = 1;
				sr_ref = 0.33;
			}
			if(temp.compare(std::string("stdev")) == 0)
			{
				local_measure = 2;
				local_measure_name = temp;
				local_measure_exp = 1;
				sr_ref = 0.35;
			}else
			{
				argi--;
			}
		}
		
		if(pname == "-lweights" || pname == "-lw" || pname == "-local-weights")
		{
				std::string option = std::string(args[argi]);				
				argi++;
				
				if(option == "circle")
				{
					lweights_name = "circle";
					lweights = 1;
					lweights_exp = true;
				}
				if(option == "gaussian")
				{
					lweights_name = "gaussian";
					lweights = 2;
					lweights_exp = true;
				}
				if(option == "none" || option == "box" || option == "constant")
				{
					lweights_name = "box";
					lweights_exp = true;
					lweights = 0;
				}
				else
				{
					argi --;
				}
		}
		
		if(pname == "-lweights-s" || pname == "-lws" || pname == "-local-weights-sigma" || pname == "-lw-sigma")
		{
			if(sl_mod)
			{
				sl = atof(args[argi]);
				sl_exp = 1;
			}
			argi ++;
		}
		// Regularization for local measure
		if( lmreg_ker_mod && ( pname.compare("-lmreg-ker") == 0 || pname.compare("-lm-reg") == 0 || pname.compare("-lm-reg-ker") == 0|| pname.compare("-lmregker") == 0 || pname == "-lmreg" ) )
		{
			std::string temp = std::string(args[argi]);
			argi++;
			
			if(temp.compare(std::string("none")) == 0 || temp.compare(std::string("off")) == 0 || temp.compare("no") == 0)
			{
				lmreg_ker = -1;
				lmreg_s = 0;
				lmreg_ker_exp = 1;
				lmreg_ker_name = temp;
			}else
			if(temp.compare(std::string("gaussian")) == 0){
				lmreg_ker = 0;
				lmreg_ker_exp = 1;
				lmreg_ker_name = temp;
				
			} else
			if(temp.compare(std::string("tukey")) == 0) {
				lmreg_ker = 1;
				lmreg_ker_exp = 1;
				lmreg_ker_name = temp;
			} else
			if(temp.compare(std::string("box")) == 0) {
				lmreg_ker = 2;
				lmreg_ker_exp = 1;
				lmreg_ker_name = temp;
			}
			else
			{
				argi--;
			}
		}
		// sigma value for local measure regularization
		if(pname.compare("-lm-reg-s") == 0 || pname.compare("-lmreg-s") == 0 || pname.compare("-lmregs") == 0 || pname.compare("-lmreg-sigma") == 0 || pname.compare("-local-measure-reg-sigma") == 0 || pname.compare("-local-measure-reg-s") == 0 || pname == "-ls" || pname == "-sl")
		{
			// If we have not set kernel, gaussian by default
			if(lmreg_ker_exp == 0)
			{
				lmreg_ker = 0;
				lmreg_ker_name = std::string("gaussian");
			}
			lmreg_s = atof(args[argi]);
			lmreg_s_exp = true;
			argi++;
		}

		if(pname.compare(std::string("-sm")) == 0)
		{
			if(sm_mod)
				sm = atof(args[argi]);
			argi ++;
		}
	
		// Check for adaptive
		if(adaptive_mod && (pname.compare("-adaptive") == 0 || pname.compare("-a") == 0))
		{
			adaptive = 1;
			adaptive_exp = 1;
		}
		if(pname.compare("-no-adaptive") == 0 || pname.compare("-adaptive-no") == 0)
		{
			adaptive = 0;
			adaptive_exp = 1;
		}
		// debug

		if(pname.compare("-d") == 0 || pname.compare("-debug") == 0)
			debug = 1;

		// Spatial Kernel
		if(sker_mod && (pname.compare("-sker") == 0 || pname.compare(std::string("-spatial-kernel")) == 0 || pname.compare("-spatial") == 0 || pname.compare("-spatialocal_measure") == 0 || pname.compare("-spatial-ker") == 0) || pname.compare("-ssker") == 0 )
		{
			std::string temp = std::string(args[argi]);
			argi ++;
	
			if(temp.compare(std::string("gaussian")) == 0){
				
				sker = 0;
				sker_exp = 1;
				
				sker_name = temp;
				
			} else
			if(temp.compare(std::string("tukey")) == 0) {	
				sker_exp = 1;
				sker = 1;		
				
				sker_name = temp;
				
			} else
			if(temp.compare(std::string("box")) == 0) {	
				sker_exp = 1;
				sker = 2;
				sker_name = temp;
			} else
			if(temp.compare(std::string("lorentz")) == 0) {
				sker_exp = 1;
				sker = 3;
				sker_name = temp;

			}
			else if(temp == "hamming-sinc" || temp == "hsinc")
			{
				sker_exp =1;
				sker = 4;
				M = atof(args[argi]);
				argi --;
				
				sker_name = temp + "_" + Patch::to_string(M);

			}
			else if(temp.compare(std::string("sinc")) == 0)
			{
				sker_exp = 1;
				sker = 5;
				M = atof(args[argi]);
				argi --;
				
				sker_name = temp + "_" + Patch::to_string(M);
				
			}
			else if(temp.compare("delta") == 0)
			{
				sker_exp = 1;
				sker = 6;
				sker_name = temp;
			}
			else{
				argi --;
			}
			
		}

		// Range kernel
		if(rker_mod && ( pname.compare(std::string("-rker")) == 0 || pname.compare(std::string("-intensity-kernel")) == 0 || pname.compare(std::string("-range-kernel")) == 0 || pname.compare("-range") == 0 ))
		{	
			
			std::string temp = std::string(args[argi]);
			argi ++;
			
			if(temp.compare(std::string("none")) == 0 || rker_name.compare(std::string("off")) == 0){	
				rker = -1;
				
				srValues.clear();
				srValues.push_back(0.0f);
				
				rker_exp = 1;
				rker_name = temp;
			} else
	
			if(temp.compare(std::string("gaussian")) == 0){	
				rker = 0; 			
				rker_exp = 1;
				rker_name = temp;
			} else
			if(temp.compare(std::string("tukey")) == 0) {	
				rker = 1;			
				rker_exp = 1;
				rker_name = temp;
			} else
			if(temp.compare(std::string("box")) == 0) {		
				rker = 2;
				rker_exp = 1;
				rker_name = temp;
			} else 
			if(temp.compare(std::string("lorentz")) == 0) {
				rker = 3;
				rker_exp = 1;
			}
			else {
				argi --;
			}
			
		}
	
		// Check for Regularization Kernel
		if(regker_mod && (pname.compare(std::string("-regker")) == 0 || pname.compare(std::string("-reg-kernel")) == 0 || pname.compare("-reg") == 0 || pname.compare("-regularization-kernel") == 0 ))
		{
			
			// Regularization kernel
			std::string temp = std::string(args[argi]);
			argi ++;
			
			if(temp.compare(std::string("none")) == 0 || temp.compare(std::string("off")) == 0 || temp.compare("no") == 0)
			{
				regker = -1;
				sreg = 0;
				regker_exp = 1;
				regker_name = temp;
			}else
	
			if(temp.compare(std::string("gaussian")) == 0){
				regker = 0;
				regker_exp = 1;
				regker_name = temp;
			} else
			if(temp.compare("tukey") == 0)
			{
				regker = 1;
				regker_exp = 1;
				regker_name = temp;
			} else
			if(regker_name.compare(std::string("box")) == 0) {
				regker = 2;
				regker_exp = 1;
				regker_name = temp;
			} else
			{
				argi--;
			}
			
		}
		
		if(pname.compare(std::string("-mker")) == 0 && mker_mod)
		{
			std::string temp = std::string(args[argi]);
			argi ++;

			if(temp.compare(std::string("none")) == 0 || temp.compare(std::string("off")) == 0)
			{
				mker = -1;
				sm = 0;				
				mker_exp = 1;
				mker_name = temp;
			}
			if(temp.compare(std::string("gaussian")) == 0)
			{
				mker = 0;
				mker_exp = 1;
				mker_name = temp;
			} else
			if(temp.compare(std::string("tukey")) == 0)
			{
				mker = 1;				
				mker_exp = 1;
				mker_name = temp;
			} else
			if(temp.compare(std::string("box")) == 0)
			{
				mker = 2;				
				mker_exp = 1;
				mker_name = temp;
			} else
			if(temp.compare(std::string("lorentz")) == 0)
			{
				mker = 3;
				
				mker_exp = 1;
				mker_name = temp;
			} else
			{
				argi--;
			}
		}
		
		// Gamma or linear input/output
		if (pname.compare("-g") == 0|| pname.compare("-gamma") == 0 )
		{
			gammacor_in = true;
			print_gamma = true;
			print_linear = false;
			gammacor_load_it = true;
		}
		if (pname.compare("-linear") == 0 || pname.compare("-no-gamma") == 0)
		{
			gammacor_in = false;
			print_gamma = false;
			print_linear = true;
			gammacor_load_it = false;
		}
		if(pname.compare("-in-linear") == 0 || pname.compare("-linear-in") == 0 || pname.compare("-linear-input") == 0 || pname.compare("-input-linear") == 0 || pname == "-lin-in" || pname == "-in-lin")
		{
			gammacor_in = false;
		}

		if(pname.compare("-out-linear-also") == 0 || pname.compare("-linear-also") == 0 || pname == "-linear-output-also" || pname == "-output-linear-also")
		{
			print_linear = true;
		}
		if(pname.compare("-out-linear-only") == 0 || pname.compare("-linear-only") == 0 || pname.compare("-out-linear") == 0 || pname.compare("-output-linear") == 0 || pname == "-linear-output" || pname == "-linear-out" || pname == "-linear-output-only" || pname == "-output-linear-only")
		{
			print_linear = true;
			print_gamma = false;
		}
		if(pname.compare("-load-it-linear") == 0)
		{
			gammacor_load_it = false;
		}
		if(pname.compare("-load-it-gamma") == 0)
		{
			gammacor_load_it = true;
		}
		if(pname.compare("-in-gamma") == 0 || pname.compare("-gamma-in") == 0 || pname == "-input-gamma" || pname == "-gamma-input")
		{
			gammacor_in = true;
		}
		if(pname.compare("-out-gamma") == 0 || pname.compare("-gamma-out") == 0 || pname.compare("-output-gamma") == 0 || pname.compare("-gamma-output") == 0)
		{
			print_gamma = true;
			print_linear = false;
		}


		// Print differences with input to PNG
		if(pname == "-print-diff-rgb" || pname == "-print-diff-rgb-gamma" || pname == "-print-diff-gamma")
		{
			print_diff_gamma = true;
		}
		if(pname == "-print-diff-linear" || pname == "-print-diff-rgb-linear")
		{
			print_diff_linear = true;
		}
		if(pname == "-print-diff-single" || pname == "-print-diff-single-gamma")
		{
			print_diff_single_gamma = true;
		}
		if(pname == "-print-diff-single-linear")
		{
			print_diff_single_linear = true;
		}
		
		if(pname == "-txt-local-measure" || pname == "-txt-lm")
		{
			txt_local_measure = true;
		}
		if(pname == "-print-lm-gamma" || pname == "-print-local-measure-gamma")
		{
			print_local_measure_gamma = true;
		}
		if(pname == "-print-lm-linear" || pname == "-print-local-measure-linear")
		{
			print_local_measure_linear = true;
		}
		
		
		// CONTRAST ENHACEMENT / REDUCTION
		
		if(pname == "-print-contrast-enhancement" || pname == "-print-ce" || pname == "-print-ce-gamma" || pname == "-print-contrast-enhancement-gamma")
		{
			make_contrast_enhacement = true;
		}
		if(pname == "-print-contrast-enhancement-linear" || pname == "-print-ce-linear")
		{
			make_contrast_enhacement_linear = true;
		}
		if(pname == "-ce" || pname == "-contrast-enhancement")
		{
			make_contrast_enhacement = true;
			ce_factor_list.clear();
			argi = read_input_list(&ce_factor_list, args, argi, nargs);
		}
		if(pname == "-ce-linear" || pname == "-contrast-enhancement-linear")
		{
			make_contrast_enhacement_linear = true;
			ce_factor_list.clear();
			argi = read_input_list(&ce_factor_list, args, argi, nargs);
		}
 
	}

	//*********************************************************************************************************************************************************************************************
	//*********************************************************************************************************************************************************************************************


	// ************************ 	VALIDATION AND FILLING OF MISSING DATA, INITIALIZATION ALSO ************************************
	// *****************************************************************************************************************************
	
	// ******************************** CONVERGENCE ***********************
	
	if(show_conv)
	calc_norms = true;
	if(conv_eps_list.size() < 1)
	{
		if(show_conv)
			conv_eps_list.push_back(conv_eps_default);
	}
	else
	{
		// Order conv_eps in reverse order.
		std::sort(conv_eps_list.begin(), conv_eps_list.end());
	}
	
	if(conv_norm_list.size() > 0)
	{
		calc_norms = true;
		calc_convergence = true;
	}
	
	// force stop on convergence if we have not set a maximum limit
	if(calc_convergence)
	{
		if(it_list.size() < 1)
		{
			force_stop_on_convergence = true;
		}
		
		if(conv_norm_list.size() < 1)
		{
			if( show_norm_list.size() > 0 )
			{
				conv_norm_list.push_back(show_norm_list.at(0));
			}
			// If we have not set conv vector but want to run until convergence, add default norm to stack
			else
				conv_norm_list.push_back(conv_norm_default);

		}
	}
	
	// We have to add the convergence criteria to the showing criteria.
	check_and_add(&show_norm_list, conv_norm_list);
	
	// We initialize the values of the index of the epsilon for comparisons
	int n_show_norms = (int) show_norm_list.size();
	for(int i = 0; i < (int)show_norm_list.size(); i++)
	{
		show_conv_values.push_back(0);
		conv_values.push_back(0);
	}

	
	// ******************* PARAMETERS ********************************************************************8
	// If ss or sr values are missing, complete with zeros
	if(ssValues.size() == 0)
	{
		if(debug) dprintf(0, "No ss values detected, added 0.0f (will not take effect)");
		ssValues.push_back(0.0f);
	}
	if(srValues.size() == 0)
	{
		if(debug) dprintf(0, "No sr values detected, added 0.0f (will not take effect)");
		srValues.push_back(0.0f);
	}
	if(scaleValues.size() == 0)
	{
		if(debug) dprintf(0, "No scale values detected. Added 1.0 as default\n");
		scaleValues.push_back(1.0f);
	}
	if(sregValues.size() == 0)
	{
		if(debug) dprintf(0, "No sreg values detected. Added 0.25 as default\n");
		sregValues.push_back(0.25f);
	}
	
	// ************************** INPUT, OUTPUT *****************************************************************
	// Inform of parameters
	if(input_name.compare("") == 0 )
	{
		printf("Input file missing.\n");
		return 0;
	}
	if(make_output_default == 1)
	{
		output_name = input_name;
	}
	// Automatic log file : input + -log.md
	if(save_log && auto_log_string){
		time_t now = time(0);		
		char* dt = ctime(&now);		
		if(debug) printf("Time: %s\n", dt);
		log_file_name = output_name + std::string(" - ") + conf_name + std::string(" - LOG - ") + std::string(dt) + std::string(".md");
		if(debug) dprintf(0, "Log file output: %s\n", log_file_name.data());
	}
		
	
	// *** Log file loading 
	if (save_log)
	{
		log_file = fopen(log_file_name.data(), "w");
		if (log_file == NULL)
		{
			save_log = false;
		}
	}

	
	if(save_log)
	{
		fprintf(log_file, "Commands: %s", commands_string.data());
	}	
	

	// ************************************ INFORM PARAMETERS TO USER *************************************************
	// *****************************************************************************************************************

	printf("\n\nALGORITHM:\t%s\n", conf_name.data());
	if (save_log) fprintf(log_file, "\nALGORITHM CONFIGURATION:\t%s (%i)\n", conf_name.data(), conf);
	
	printf("\nPARAMETERS:\n");
	
	printf("\nInput: \t%s\n", input_name.data());
	if(gammacor_in)
		printf("Input will be gamma corrected.\n");
	else
		printf("Input WONT	be gamma corrected.\n");
	
	if(save_log)
	{
		fprintf(log_file, "\nInput:\t%s\n", input_name.data());
	}
	
	//printf("\nInput PNG image:%s.png\n", input_name.data());
	//if (save_log) fprintf(log_file, "\nInput image:%s.png\n", input_name.data());
	
	//printf("Input scaling factor: %.2f\n", scale);
	printf("Output image prefix: %s\n", output_name.data());
	if (save_log) fprintf(log_file, "Output image: %s\n", output_name.data());
	
	printf("\nSpatial kernel:\t %s", sker_name.data());
	if(save_log) fprintf(log_file, "\nSpatial kernel:\t %s", sker_name.data());
	
	printf("\n\tss values: \t"); printfVector(ssValues); printf("\n");
	if(save_log)
	{
		fprintf(log_file, "\nss values: \t");
		fprintfVector(log_file, ssValues);
		fprintf(log_file, "\n");
	}

	printf("\nRegularization kernel:\t %s\n", regker_name.data());
	if(save_log) fprintf(log_file, "Regularization kernel:\t %s", regker_name.data());

	printf("\tsreg values: \t"); printfVector(sregValues); printf("\n");
	if(save_log)
	{
		fprintf(log_file, "\nsreg values: \t");
		fprintfVector(log_file, sregValues);
		fprintf(log_file, "\n");
	}
	
	printf("\nRange kernel:\t %s", rker_name.data());
	if (save_log) fprintf(log_file, "Range kernel:\t %s", rker_name.data());
	
	printf("\n\tsr values: \t"); printfVector(srValues); printf("\n");
	if(save_log)
	{
		fprintf(log_file, "\nsr values: \t");
		fprintfVector(log_file, srValues);
		fprintf(log_file, "\n");
	}
	
	printf("\nInput Scaling values: \t"); printfVector(scaleValues); printf("\n");
	if(save_log )
	{
		fprintf(log_file, "Input scaling values: \t");\
		fprintfVector(log_file, scaleValues);
		fprintf(log_file, "\n");
	}
	
	printf("Scale back output: %s\n", scale_back_name.data());
	if(save_log) fprintf(log_file, "Scale back output: %s\n", scale_back_name.data());
	

	
	printf("\nAdaptive algorithm:\t %s (%i)\n", adaptive_name.data(), adaptive);
	if(save_log) fprintf(log_file, "Adaptive algorithm:\t %s (%i)\n", adaptive_name.data(), adaptive);
	printf("\tInfsr:\t%.4f\n", infsr);
	
	if(adaptive)
	{
		printf("\tLocal Measure:\t %s", local_measure_name.data());
		if (save_log) fprintf(log_file, "\tLocal measure calculation kernel for adaptiveness:\t %s\n", local_measure_name.data());
		if(!sl_exp)
			printf("\n\tLocal measure weights:\t%s, sl = auto\n", lweights_name.data());
		else
			printf("\n\tLocal measure weights: \t%s, sl = %f", lweights_name.data(), sl);
		if (save_log) fprintf(log_file, "\tLocal measure weights:\t%s, sl = %.4f\n", lweights_name.data(), sl);
		printf("\tLocal measure regularization: \t%s, lmreg-s = %.4f\n", lmreg_ker_name.data(), lmreg_s);
		if (save_log) fprintf(log_file, "\tLocal measure regularization: \t%s, lmreg-s = %.4f\n", lmreg_ker_name.data(), lmreg_s);
	}
	//printf("Median kernel:\t %s, ms = %.4f\n", mker_name.data(), sm);	
	// if we are not spedified iterations to print, just last one
	if(print_exp == 0 && calc_norms == false)
	{
		it_list.push_back(nit);
	}
	printf("\nITERATIONS to print:\t"); printfVector(it_list); printf("\n");
	if(save_log)
	{
		fprintf(log_file, "Iterations to print:\t");
		fprintfVector(log_file, it_list);
		fprintf(log_file, "\n");
	}
	if(save_txt)
	{
		printf("Txt versions will also be saved in linear space in [0,1] range.\n");
		if(save_log)
			fprintf(log_file, "Txt versions will also be saved in linear space in [0,1] range.\n");
	}
	
	if(gammacor_in == false) printf("Input will be read in linear space. (No gamma correction)\n");
	else printf("Input will be gamma corrected.\n");
	if(print_linear) printf("Output will be written in linear space (No gamma correction)\n");
	if(print_gamma) printf("Output will be gamma corrected.\n");
	if(save_log)
	{
		if(gammacor_in == false) fprintf(log_file, "Input will be read in linear space. (No gamma correction)\n");
		if(print_gamma == false) fprintf(log_file, "Output will be written in linear space (No gamma correction)\n");
	}
	
	if(load_it)
	{
		printf("Loading iteration %i from %s\n", it_from, it_from_image_name.data());
		if(save_log) fprintf(log_file, "Loading iteration %i from %s\n", it_from, it_from_image_name.data());
	}
	if(calc_norms)
	{
		printf("\nConvergence criteria:\t");
		printfVector(conv_norm_list);
		printf("\n");
		
		if (save_log)
		{
			fprintf(log_file, "\nConvergence criteria:\t" );
			fprintfVector(log_file, conv_norm_list);
			fprintf(log_file, "\n");
		}
		
		printf("Convergence for eps :"); printfVector(conv_eps_list); printf("\n");
		if(save_log)
		{
			 fprintf(log_file, "Convergence norm epsilon : "); fprintfVector(log_file, conv_eps_list); fprintf(log_file, "\n"); 
		}
		if (force_stop_on_convergence)
		printf("* Algorithm will STOP if convergence is reached.\n");
		if(save_log) fprintf(log_file, "* Algorithm will STOP if convergence is reached because of -force-stop.\n");
		
		printf("* Printing convergence iterations by default.\n");
		if(save_log) fprintf(log_file, "* Printing convergence iterations by default.\n");
		
		printf("\nShowing info for norms : ");
		printfVector(show_norm_list);
		printf("\n");
		
		if(show_conv) printf("Showing convergence for  eps = "); printfVector(conv_eps_list);
		if (print_alt_conv) printf("Printing Convergence Iterations for these norms.\n");
		
		if(save_log)
		{
			fprintf(log_file, "\nShowing info for norms : ");
			fprintfVector(log_file, show_norm_list);
			fprintf(log_file, "\n");
			
			if(show_conv) fprintf(log_file, "Showing convergence for  eps = "); fprintfVector(log_file, conv_eps_list);
			if (print_alt_conv) fprintf(log_file, "Printing Convergence Iterations for these norms.\n");		
		}
		
	}
	else
	{
		printf("\nConvergence metric:\tnone\n");
		if(save_log) fprintf(log_file, "Convergence metric:\tnone\n");
	}

	if(save_log)
		printf("\nlog file will be saved to %s\n", log_file_name.data());
		
		
		
	// ************* INPUT LOADING ********************************
	// ****************************************************************
		
	if(debug) printf("\n LOADING INPUT AND INITIALIZATION OF CPU / GPU ARRAYS .. \n");
	
	int H, W, nchannels, nch_no_alpha;
	int gH, gW, gnchannels, gnch_no_alpha;
	unsigned char color_type, bit_depth;
	unsigned char gcolor_type, gbit_depth;
	//pixel buffer
	unsigned char *pixels = NULL;
	unsigned char *gpixels = NULL;
	
	printf("Reading input image f ... ");
	//read_png(args[1], H, W, nchannels, pixels, color_type, bit_depth, nch_no_alpha);
	read_png((input_name + ".png").data(), H, W, nchannels, pixels, color_type, bit_depth, nch_no_alpha);
	if(bit_depth != 8)
	{
		dprintf(0, "Reading Input: bit depth of %i is not supported yet. Aborting ... \n", bit_depth);
		return 0;
	}
	if(load_it)
	{
		read_png((it_from_image_name + ".png").data(), gH, gW, gnchannels, gpixels, gcolor_type, gbit_depth, gnch_no_alpha);
		if(gbit_depth != 8)
		{
			dprintf(0, "Reading Input Iteration: bit depth of %i is not supported yet. Aborting ... \n", gbit_depth);
			return 0;
		}
			// Images have to match
		if(gH != H || gW != W || gnchannels != nchannels || gcolor_type != color_type || gnch_no_alpha != nch_no_alpha)
		{
			dprintf(0, "Input and Iteration images do not match.\n");
			return 0;
		}
	}

	printf("Image width: %i, Image height: %i, Number of channels: %i  ",H, W, nchannels);
	if(debug) printf("done.\n");
	
	type RGB_REF = pow(2, bit_depth) -1;
	
	if(debug) printf("Creating arrays of channels for f and g and get out gamma correction if wanted ... ");
	
	//HOST: Create arrays for each channel, alpha channel will not be filtered
	type *host_f = new type[nch_no_alpha * H * W];
	type *host_g = new type[nch_no_alpha * H * W];
	type *host_temp = new type[nch_no_alpha * H * W];	// Will save host iterations as well as norms to show
	
	if(debug) dprintf(0, "Initializing f in Host ... ");
	for(int ch = 0; ch< nch_no_alpha ; ch++){
		unsigned char *p = (pixels + ch);
		//fill channel i for f
		for(int i = 0; i< H*W; i++){
		
			type val = ((type)(*p))/RGB_REF;
			// Take out gamma correction
			if(gammacor_in == true){
				if(val <= 0.04045)
					val = val/12.92f;
				else	
					val = pow( (val + 0.055f)/1.055, 2.4f);
			}
			// Initialize images in Host
			host_f[ch * W * H + i] = val;			
			p += nchannels;
		}
	}
	if(debug) dprintf(0, "done.\n");
	
	// Save txt slices for original function also if for comparisons
	
	if(h_slice_list.size() > 0)
	{
		printf("Saving H slices for input image f ... \n");
		for(int index = 0; index < (int)h_slice_list.size(); index++)
		{
			int i = h_slice_list.at(index);
			std::string slice_file_name = input_name + "-hslice_" + Patch::to_string(i) + ".txt";
			FILE *slice_file = fopen(slice_file_name.data(), "w");
			 
			for(int j = 0; j < W; j++)
			{
				for(int ch = 0; ch < nch_no_alpha; ch ++)
				{
					fprintf(slice_file, "%.8f\t", host_f[ch * W * H + i * W + j]);
				}
				fprintf(slice_file, "\n");
			}
			fclose(slice_file);
			dprintf(0, "Input H slice %d saved in %s.\n", i, (slice_file_name ).data());						
		}
	}
	
	if(v_slice_list.size() > 0)
	{
		printf("Saving H slices for input image f ... \n");
		for(int index = 0; index < (int)v_slice_list.size(); index++)
		{
			int j = v_slice_list.at(index);
			std::string slice_file_name = input_name + "-vslice_" + Patch::to_string(j) + ".txt";
			FILE *slice_file = fopen(slice_file_name.data(), "w");
			 
			for(int i = 0; i < W; i++)
			{
				for(int ch = 0; ch < nch_no_alpha; ch ++)
				{
					fprintf(slice_file, "%.8f\t", host_f[ch * W * H + i * W + j]);
				}
				fprintf(slice_file, "\n");
			}
			fclose(slice_file);
			dprintf(0, "Input V slice %d saved in %s.\n", j, (slice_file_name ).data());						
		}
	}
	printf("\n");	

	// **** Intialization of data for algorithms
	
	if(debug) dprintf(0, "Initializing g in Host ...");
	for(int ch = 0; ch< nch_no_alpha ; ch++){
		unsigned char *p = (gpixels +ch);
		//fill channel i for f
		for(int i = 0; i< H*W; i++){	
			// Initialize images in Host		
			if(load_it == false)
			{
				if(conf == 3 )
				{
					host_g[ch * W * H + i] = host_f[ch * W * H + i];	// bilateral filter starts with f
				}
				else
				{
					host_g[ch * W * H + i] = (type) 0;
				}
			}
			else
			{
				type val = ((type)(*p))/RGB_REF;
			
				// Take out gamma correction
				if(load_it_gammacor){
					if(val <= 0.04045)
						val = val/12.92f;
					else	
						val = pow( (val + 0.055f)/1.055, 2.4f);
				}
				host_g[ch * W + H + i] = val;
			}
			p += nchannels;
		}
	}
	if(debug) dprintf(0, "done.\n");

	// Scale sr value to have a fair RGF comparison

	type f_max = max_rgb_norm2<type>(host_f, H, W, nch_no_alpha);
	type f_min = min_rgb_norm2<type>(host_f, H, W, nch_no_alpha);
	type f_range = sqrt(f_max) - sqrt(f_min);
	
	
	if(debug) printf("GPU INITIALIZATION\n");
	if(debug) printf("Getting available GPU memory ...\n");
	size_t free;
	size_t total;
	cudaMemGetInfo(&free, &total);
	
	if (debug) printf("There are %lu bytes available of %lu\n", free, total);
	long unsigned int needgpu = 4 * nch_no_alpha * H * W* sizeof(type) +  H * W* sizeof(type);
	if(debug) printf("allocating images memory in device, need at least %lu free bytes on GPU ... \n", needgpu);
	if(free < needgpu)
		if(debug) printf("Not enough available memory on GPU. There can be errors.\n");
		
	// DEVICE memory allocation
	if(debug) printf("allocating images memory in device ... ");
	type *dev_f, *dev_g, *dev_g_last;
	type *dev_stdev, *dev_temp;
	type *dev_temp2;
	if(calc_norms){
		cudaMalloc((void**)&dev_g_last, nch_no_alpha * H * W * sizeof(type));
		gpuErrChk( cudaPeekAtLastError() );
	}
	if(adaptive)
	{
		cudaMalloc((void**)&dev_stdev, H * W * sizeof(type));
		gpuErrChk( cudaPeekAtLastError() );
	}
	cudaMalloc((void**)&dev_temp, nch_no_alpha * H * W * sizeof(type));
	gpuErrChk( cudaPeekAtLastError() );
	
	cudaMalloc((void**)&dev_temp2, nch_no_alpha * H * W * sizeof(type));
	gpuErrChk( cudaPeekAtLastError() );
	
	cudaMalloc((void**)&dev_f, nch_no_alpha * H * W * sizeof(type));
	gpuErrChk( cudaPeekAtLastError() );
	cudaMalloc((void**)&dev_g, nch_no_alpha * H * W * sizeof(type));
	gpuErrChk( cudaPeekAtLastError() );
	
	if(debug) printf("done.\n");
	
	type *host_local_measure = NULL;
	if(txt_local_measure || print_local_measure_gamma || print_local_measure_linear)
	{
		host_local_measure = new type[H * W];
	}
	
	dim3 ggridsize = dim3(ceil((type)W/gbsX), ceil((type)H/gbsY), 1);
	dim3 gblocksize = dim3(gbsX, gbsY,1);
	
	// ******************************************************************************************************************************
	//  ************* FILTERING 
	// ******************************************************************************************************************************
	
	bool print_it = false; // True if iteration is to be printed	
	output_name = output_name + "-" + conf_name;
	
	// LOOPS:
	if(debug) dprintf(0, "Starting SCALEloop \n");
	for(int i = 0; i < (int)scaleValues.size(); i++)
	{
	
		type scale = scaleValues.at(i);
		if(debug) dprintf(0, "scale = %.2f\n", scale);

		// Scale range
		f_range *= scale;
		if(f_range == 0) f_range = 1.0;
		
		if(debug) dprintf(0,"Copying data into device ... ");
		for(int ch = 0; ch<nch_no_alpha; ch++){
			cudaMemcpy(dev_f + ch*H*W , host_f + ch * H * W, H*W*sizeof(type), cudaMemcpyHostToDevice);
			gpuErrChk( cudaPeekAtLastError() );		
		}
		if(debug) dprintf(0, "Scaling input for scale = %.2f", scale);
		for(int i = 0; i<nch_no_alpha; i++)
		{
			scale_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_temp + i * H * W,  scale, H, W);
			gpuErrChk( cudaPeekAtLastError() );
		}
		std::swap(dev_f, dev_temp);
		

		
	if(debug) dprintf(0, "\nStarting SS loop");
	for(int i = 0; i < (int)ssValues.size(); i++)
	{		
		type ss = ssValues.at(i);
		if(debug) dprintf(0, "ss = %.2f\n", ss);
	
		if(!sl_exp) sl = ss;
			
		// Calculate size of regions
		switch(sker)
		{
			case 0:
				sskh = ceil(3.0f * ss);
				sskw = ceil(3.0f * ss);
				break;
			case 1:
				sskh = ceil(ss);
				sskw = ceil(ss);
				break;
			case 2:
				sskh = ceil(ss);
				sskw = ceil(ss);
				break;
			case 3:
				sskh = ceil(ss);
				sskw = ceil(ss);
				break;
			case 4:
				sskh = ceil(0.5f * M / ss);
				sskw = ceil(0.5f * M / ss);
			case 5:
				sskh = ceil( 0.5f * M /  ss);
				sskw = ceil( 0.5f * M / ss);
				break;
			default:
				break;
		}
		if(debug) dprintf(0, "ss Region for gpu: sskh = %i, sskw = %i\n", sskw, sskw);
	
	// Loop for regularization values (fraction of ss)
	if(debug) dprintf(0, "\nStarting SREG loop ...");		
	for(int i = 0; i < sregValues.size(); i++)
	{
		type sreg = sregValues.at(i);
		if(debug) dprintf(0, "sreg = %.2f\n", sreg);

		switch(regker)
		{
			case 0:
				regkh = ceil(3.0f * sreg * ss);
				regkw = ceil(3.0f * sreg * ss);
				break;
			case 1:
				regkh = ceil(sreg);
				regkw = ceil(sreg);
				break;
			case 2:
				regkh = ceil(sreg);
				regkw = ceil(sreg);
				break;
			case 3:
				regkh = ceil(sreg);
				regkw = ceil(sreg);
				break;
			default:
				break;
		}
		if(debug) dprintf(0, "sreg region for gpu: regkh = %i, regkw = %i\n", regkh, regkw);
		
		// Sigma regularization for local measure = sigma of local measure kernel by default
		if(lmreg_ker_exp && !lmreg_s_exp)
		{
			lmreg_s = sl;
		}
		switch(lmreg_ker)
		{
			case 0:
				lmreg_kh = ceil(3.0f * lmreg_s);
				lmreg_kw = ceil(3.0f * lmreg_s);
				break;
			case 1:
				lmreg_kh = ceil(lmreg_s);
				lmreg_kw = ceil(lmreg_s);
				break;
			case 2:
				lmreg_kh = ceil(lmreg_s);
				lmreg_kw = ceil(lmreg_s);
				break;
			case 3:
				lmreg_kh = ceil(lmreg_s);
				lmreg_kw = ceil(lmreg_s);
				break;
			default:
				break;
		}	
		if(debug) dprintf(0, "lmreg region for gpu: lmreg_kh = %i, lmreg_kw = %i\n", lmreg_kh, lmreg_kw);
		switch(lweights)
		{
			// exponential weights
			case 1:
				lkh = ceil(3.0f * sl);
				lkw = ceil(3.0f * sl);
				break;
			// circle characteristic weights
			case 0:
				lkh = ceil(3.0f * sl);
				lkw = ceil(3.0f * sl);
				break;
			default:
				lkh = ceil(3.0f * sl);
				lkw = ceil(3.0f * sl);
				break;
		}
		if(debug) dprintf(0, "lweights region for gpu: lkh = %i, lkw = %i\n", lkh, lkw);

	if(debug) dprintf(0, "\nStarting SR loop ...");		
	for(int i = 0; i < (int)srValues.size(); i++)
	{		
		type sr = srValues.at(i);
		
		// ************** PROCESS for given parameters ***************
		
		// Start convergence values over
		for(int i = 0; i < (int)show_norm_list.size(); i++)
		{
			show_conv_values.at(i) = conv_eps_list.size();
			conv_values.at(i) = conv_eps_list.size();	
		}

		for(int i = 0; i < (int)show_norm_list.size(); i++)
		{
			if(isInVector(conv_norm_list, show_norm_list.at(i)) < 0)
			{
				conv_values.at(i) = 0;
			}
		}
		
		dprintf(0, "\n\n\tWorking for ss = %.2f, sr = %.2f, sreg = %.2f, scale = %.2f\n", ss, sr, sreg, scale);
		if(save_log) fprintf(log_file, "\n\n\tWorking for ss = %.2f, sr = %.2f, scale = %.2f\n", ss, sr, scale);
		
		//Initialize dev_g according to conf
		if(debug) dprintf(0, "\nIntializing dev_g in device for conf %s (%i) (copying from g_host)... ", conf_name.data(), conf);
		cudaMemcpy(dev_g, host_g, nch_no_alpha * H * W * sizeof(type), cudaMemcpyHostToDevice);				
		if(debug) dprintf(0, "done.");	
	
		// Perform iterations
		bool run = true;
		
		int it = it_from;
		
		nit = 0;
		if(it_list.size() > 0)
			nit = it_list.at(maxInVector(it_list));
		// Check if we still want to run iterations
		run = (calc_convergence) || (it < nit);
		
		// Run iterations
		bool converged = false;
		auto start = std::chrono::high_resolution_clock::now();
		while(run)
		{
			// Iteration info
			it++;
			dprintf(0, "\nIt %i\t", it);
			if(save_log) fprintf(log_file, "\nIt %i\t", it);
			
			cudaDeviceSynchronize();
			
			// host_g_last copies g data to calculate l2loc norm
			if(calc_norms)
			{
				cudaMemcpy(dev_g_last, dev_g, H * W * nch_no_alpha * sizeof(type), cudaMemcpyDeviceToDevice);
				gpuErrChk( cudaPeekAtLastError() );
				cudaDeviceSynchronize();
			}
	
			// Adaptive Calculation in dev_stdev
			if(adaptive && local_measure >= 0)
			{
				dprintf(0, "lm ");
				if(save_log) fprintf(log_file, "a ");
				
				// Calculate local measure values
				switch(local_measure)
				{
					case 0:	// Max as local measure
					
						for(int i = 0; i < nch_no_alpha; i++)
						{
							// stdv is calculated in 3d using euclidean metric in rgb space
							//subs_kernel_rgb<type><<<ggridsize, gblocksize>>>(dev_f, dev_g, devtemp, nch_no_alpha, H, W); -> dev_temp_i
							subs_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_g + i * H * W, dev_temp + i * H * W, H, W);
							gpuErrChk( cudaPeekAtLastError() );
							
							// Calculate mean value for each channel. -> dev_temp2
							//local_mean_kernel<type><<<ggridsize, gblocksize>>>(dev_temp + i * H * W, dev_temp2 + i * H * W, H, W, sl, lweights, lkh, lkw);
							//gpuErrChk( cudaPeekAtLastError() );
				
						}
						//centered_max_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_temp2, dev_stdev, H, W, nch_no_alpha, sl, lweights, lkh, lkw);
						max_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_stdev, H, W, nch_no_alpha, sl, lweights, lkh, lkw);
						gpuErrChk( cudaPeekAtLastError() );						
						cudaDeviceSynchronize();
						
						break;
			
					case 1:	// Mean as local measure
					
						for(int i = 0; i < nch_no_alpha; i++)
						{
							// stdv is calculated in 3d using euclidean metric in rgb space
							//subs_kernel_rgb<type><<<ggridsize, gblocksize>>>(dev_f, dev_g, devtemp, nch_no_alpha, H, W); -> dev_temp_i
							subs_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_g + i * H * W, dev_temp + i * H * W, H, W);
							gpuErrChk( cudaPeekAtLastError() );
							
							// Calculate mean value for each channel. -> dev_temp2
							//local_mean_kernel<type><<<ggridsize, gblocksize>>>(dev_temp + i * H * W, dev_temp2 + i * H * W, H, W, sl, lweights, lkh, lkw);
							//gpuErrChk( cudaPeekAtLastError() );
				
						}
						mean_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_stdev, H, W, nch_no_alpha, sl, lweights, lkh, lkw);
						//centered_mean_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_temp2, dev_stdev, H, W, nch_no_alpha, sl, lweights, lkh, lkw);
						gpuErrChk( cudaPeekAtLastError() );						
						cudaDeviceSynchronize();
						
						break;
					

					
					default: // 2 or default: Standard deviation as local measure
					
						for(int i = 0; i < nch_no_alpha; i++)
						{
							// stdv is calculated in 3d using euclidean metric in rgb space
							//subs_kernel_rgb<type><<<ggridsize, gblocksize>>>(dev_f, dev_g, devtemp, nch_no_alpha, H, W); -> dev_temp_i
							subs_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_g + i * H * W, dev_temp + i * H * W, H, W);
							gpuErrChk( cudaPeekAtLastError() );
							
							// Calculate mean value for this channel. -> dev_stdev
							local_mean_kernel<type><<<ggridsize, gblocksize>>>(dev_temp + i * H * W, dev_stdev, H, W, sl, lweights, lkh, lkw);
							gpuErrChk( cudaPeekAtLastError() );
					
							// Calculate the mean of the squared distances to each mean value, -> dev_temp2_i
							squared_centered_mean_kernel<type><<<ggridsize, gblocksize>>>(dev_temp + i * H * W, dev_stdev, dev_temp2 + i * H * W,  H, W, sl, lweights, lkh, lkw);
							gpuErrChk( cudaPeekAtLastError() );
							
							
							// Check for local measure local_measure: stdev, max or mean, each one can have differente weights lweights
							// Notice it is calculated on  a (2 lkh + 1) x (2 lkw + 1) neighboorhood
							// local_measure 0 = max 1 = mean 2 = stdev
				
						}
						sum_channels_kernel<type><<<ggridsize, gblocksize>>>(dev_temp2, dev_stdev, H, W, nch_no_alpha);
						gpuErrChk( cudaPeekAtLastError() );
						
						// Normalize
						scale_in_place_kernel<type><<<ggridsize, gblocksize>>>(dev_stdev, 1.0 / ((type)nch_no_alpha), H, W);
						gpuErrChk( cudaPeekAtLastError() );

						sqrt_in_place_kernel<type><<<ggridsize, gblocksize>>>(dev_stdev, H, W);
						gpuErrChk( cudaPeekAtLastError() );
						
						cudaDeviceSynchronize();
						
						break;
					
					
				}
				
				// dev_temp is free now
				
				//Regularize local measure values if wanted
				if(lmreg_ker >= 0 && lmreg_s >0)
				{
					dprintf(0, "lmreg ");
					if(save_log) fprintf(log_file, "lmreg ");
					
			
					convolution_kernel<type><<<dim3(ceil((type)W/blockConvW), ceil((type)H/blockConvH), 1), dim3(blockConvW, blockConvH,1)>>>(dev_stdev, dev_temp,  H, W, lmreg_s, lmreg_ker, lmreg_kh, lmreg_kw);
					gpuErrChk( cudaPeekAtLastError() );
					
					cudaMemcpy(dev_stdev, dev_temp, H * W * sizeof(type), cudaMemcpyDeviceToDevice);
					gpuErrChk( cudaPeekAtLastError() );
		
				}
				cudaDeviceSynchronize();
				
				//dev_temp is free now
			}
			
			// Regularization in dev_g
			if(regker >= 0 && sreg > 0)
			{
				dprintf(0, "reg ");
				if(save_log) fprintf(log_file, "reg ");

				for(int i = 0; i<nch_no_alpha; i++){			
					convolution_kernel<type><<<dim3(ceil((type)W/blockConvW), ceil((type)H/blockConvH), 1), dim3(blockConvW, blockConvH,1)>>>(dev_g + i * H * W, dev_temp + i * H * W,  H, W, sreg * ss , regker, regkh, regkw);
					gpuErrChk( cudaPeekAtLastError() );
				}
				std::swap(dev_g, dev_temp);
				// dev_temp is free now
			}
			cudaDeviceSynchronize();
			
			// Joint Bilateral Filter
			dprintf(0, "b\t");
			for(int i = 0; i<nch_no_alpha; i++)
			{	
				type* devf = dev_f;
				if(conf == 3)
				{
					// Bilateral algorithm do not work on original f
					devf = dev_g;
				}
				if (adaptive)
				{
					trilateral_kernel_rgb<type><<<dim3(ceil((type)(W)/blockBilW), ceil((type)(H)/blockBilH),1), dim3(blockBilW, blockBilH,1)>>>(devf + i * H * W, dev_g, dev_temp + i * H * W, nch_no_alpha, H, W, ss, sr / sr_ref, dev_stdev, infsr, sm, sker, rker, mker, sskh, sskw);
				}
				else
				{
					trilateral_kernel_rgb<type><<<dim3((int)(W/blockBilW) + 1, (int)(H/blockBilH) + 1,1), dim3(blockBilW, blockBilH,1)>>>(devf + i * H * W, dev_g, dev_temp + i * H * W, nch_no_alpha, H, W, ss, sr * f_range, sm, sker, rker, mker, sskh, sskw);
				}
				gpuErrChk( cudaPeekAtLastError() );	
		
			}
			// dev_g to store output:
			std::swap(dev_g, dev_temp);
			// dev_temp is free now

			// If we want to check convergence or norm between iterations
			if(calc_norms)
			{
				if(debug) dprintf(0, "Calculating norms for it %i\n", it);
				//for every norm in norm_vector we have to calculate to show info to user. Maybe we could optimize so many comparisons
				for(int i = 0; i < (int)show_norm_list.size(); i++)
				{
					std::string norm_string = show_norm_list.at(i);
					type norm_val = 0.0f;

					if(norm_string.compare("l2") == 0)
					{
						// Calculate values
						dist2_rgb_kernel<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, nch_no_alpha);
						gpuErrChk( cudaPeekAtLastError() );
						
						// copy this values to host to calculate max value
						if(debug) dprintf(0, "\nDev to Host to check stopping criteria:");				
						cudaMemcpy( host_temp, dev_temp, H * W * sizeof(type), cudaMemcpyDeviceToHost );
						gpuErrChk( cudaPeekAtLastError() );
						
						norm_val = mean<type>(host_temp, H, W);
						
						
					}
					else if(norm_string.compare("l2loc") == 0)
					{
						// Calculate L2loc norm between last and current iteration	and stores in dev_temp (first channel)			
						l2loc2_kernel_rgb<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, nch_no_alpha, sskh, sskw);
						gpuErrChk( cudaPeekAtLastError() );
						// copy this values to host to calculate max value
						if(debug) dprintf(0, "\nDev to Host to check stopping criteria:");				
						cudaMemcpy( host_temp, dev_temp, H * W * sizeof(type), cudaMemcpyDeviceToHost );
						gpuErrChk( cudaPeekAtLastError() );
					
						// Calculate max values
						norm_val = max_abs<type>(host_temp, H, W);
					}
					else if(norm_string.compare("max") == 0)
					{
						l2loc2_kernel_rgb<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, nch_no_alpha, 0, 0);
						//dist2_rgb_kernel<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, nch_no_alpha);
						gpuErrChk( cudaPeekAtLastError() );
						
						// copy this values to host to calculate max value
						if(debug) dprintf(0, "\nDev to Host to check stopping criteria:");				
						cudaMemcpy( host_temp, dev_temp, H * W * sizeof(type), cudaMemcpyDeviceToHost );
						gpuErrChk( cudaPeekAtLastError() );
						
						norm_val = max_abs<type>(host_temp, H, W);
						
					}
					
					dprintf(0, "  %s = %.8f", norm_string.data(), sqrt(norm_val));
					if(save_log) fprintf(log_file, "  %s = %.16f", norm_string.data(), sqrt(norm_val));
					
					if(show_conv_values.at(i) > 0)
					{
						type eps = conv_eps_list.at(show_conv_values.at(i) -1 );
						if( norm_val <= eps * eps * nch_no_alpha)
						{	
							dprintf(0, "\n * Convergence at Iteration %i with %s = %.16f for eps = %.8f\n", it, norm_string.data(), sqrt(norm_val), eps);
							if(save_log) fprintf(log_file, "\n * Convergence at Iteration %i with %s = %.16f for eps = %.16f\n", it, norm_string.data(), sqrt(norm_val), eps);
							
							if(print_alt_conv) print_it = true;
							
							show_conv_values.at(i) -= 1;
							
							if(conv_values.at(i) > 0)
							{
								conv_values.at(i) -= 1;
								print_it = true;
							}
						}					
					}

				}
				if(sumVector(conv_values) == 0)
				{
					dprintf(0,"converged = true\n");
					converged = true;
					if( stop_showing_on_convergence == false) calc_norms = true;
				}
			}
			// Print iterations in it_list vector
			if (print_all)
			{
				print_it = true;
			}
			else if(it > 0 && isInVector(it_list, it, 0) >= 0)
			{
				if (debug) dprintf(0, "\nIteration %i is in printing vector.", it);
				print_it = true;
			}
			// If we achieve max it then print result
			if(it > 0 && it == max_it)
			{
				dprintf(0, "\nWe arrived to max iteration %i", max_it);
				print_it = true;
			}
			// If we have printing conditions, print iteration and related options
			if(print_it)
			{
				auto stop = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
				dprintf(0,"\tExecution time:\t%ld microseconds\n", duration.count());		
				// Make output generic name
				std::string out_name = output_name;

				// Add adapt_sr info to RGF
				if(adapt_sr && adaptive == false)
				{
					out_name += "_asr";
				}

				if(rker_exp)
				{
					out_name += "-rker-" + rker_name;
				}
				if(sr > 0)
				{
					out_name += "-sr-" +  Patch::to_string_f(sr, 2);
				}
				if(sker_exp)
				{
					out_name += "-sker-" + sker_name;
				}		
				if(ss > 0)
				{
					if(xic_exp == false)
					{
						out_name += "-ss-" +  Patch::to_string_f(ss, 2);
					}
					else
					{
						out_name += "xic_1_" + Patch::to_string_f(ss / calibration_xicxss,2);
					}
				}
				if(sreg_exp) out_name += "-sreg_" + Patch::to_string_f(sreg, 2);
				// Add Adaptive Filter info
				if (adaptive)
				{
					out_name += "-lm_" + local_measure_name;
					out_name += "_" + lweights_name + "_" + Patch::to_string_f(sl, 2);
				
					if(lmreg_ker_exp)
					{
						out_name += "-lmreg_" + lmreg_ker_name + "_" + Patch::to_string_f(lmreg_s, 2);
					}
				}
				
				// Add iteration info
				out_name += "-nit-" + Patch::to_string(it);

				// Add scaling info
				if (scale_exp)
				{
					out_name += "-scale" ;
					if (scale_back)
					{
						out_name += "_sb";
					}
					out_name += "-" + Patch::to_string_f(scale, 2);
				}
				
				// Add infsr info
				if(infsr_exp == true)
				{
					out_name += "-infsr_" + Patch::to_string_f(infsr, 2);
				}
				
				// Copy back to Host
				if(debug) dprintf(0, "\nGetting back result to host ... ");
				for(int i = 0; i < nch_no_alpha; i++)
				{		
					cudaMemcpy( host_temp + i * H * W, dev_g + i * H * W, H * W * sizeof(type), cudaMemcpyDeviceToHost );
					gpuErrChk( cudaPeekAtLastError() );
					
				}

				if(adaptive && (txt_local_measure || print_local_measure_gamma || print_local_measure_linear))
				{
					cudaMemcpy(host_local_measure, dev_stdev, H * W * sizeof(type), cudaMemcpyDeviceToHost);
					gpuErrChk(cudaPeekAtLastError());
				}
				
				if(adaptive && txt_local_measure)
				{
					dprintf(0, "\nSaving LOCAL MEASURE TXT ...");
					std::string file_name = out_name + "-lmeasure.txt";				
					
					FILE *file = fopen(file_name.data(), "w");
					for(int i = 0; i < H; i++)
					{
						for(int j = 0; j < W; j++)
						{
							fprintf(file, "%.8f\t", host_local_measure[i * W + j]);
						}
						fprintf(file, "\n");
					}
					fclose(file);
					dprintf(0, "\n\t> %s saved.", file_name.data());
				}
				if(debug) dprintf(0, "done.\n");
				// Print local measure to png with gamma correction
				if(adaptive && print_local_measure_gamma)
				{
					dprintf(0, "\nPrinting LOCAL MEASURE with GAMMA correction ...");
					std::string file_name = out_name + "-lm_gamma.png";
					unsigned char* pixels = new unsigned char[H * W];
					//unsigned char * pix = new unsigned char[H * W * nchannels];
					unsigned char *p = pixels;
				
					//Change first color channels
					for(int i = 0; i < H*W; i++)
					{
							type val = host_local_measure[i];
							
							if(val <= 0.0031308)
								val = 12.92 * val;
							else
								val = 1.055 * pow( val, 1.0/2.4 ) - 0.055;
								
							*p = (unsigned char)(val * RGB_REF);
							p++;
					}
					if(debug) dprintf(0, "done.\n");
					
					save_image(file_name.data(), pixels, H, W, 1, 0, 8);
					dprintf(0, "\n\t> %s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\nLocal Measure with Gamma > %s saved.", file_name.data());
					}
					delete[] pixels;
				
				}
				
				if(adaptive && print_local_measure_linear)
				{
					// Save PNG image
					dprintf(0, "\nPrinting LOCAL MEASURE in LINEAR space ...");
					std::string file_name = out_name + "-lm_linear.png";
					unsigned char* pixels = new unsigned char[H * W];
					//unsigned char * pix = new unsigned char[H * W * nchannels];
					unsigned char *p = pixels;
				
					//Change first color channels
					for(int i = 0; i < H*W; i++)
					{
							type val = host_local_measure[i];
							
							*p = (unsigned char)(val * RGB_REF);
							p++;
					}
					if(debug) dprintf(0, "done.\n");

					
					save_image(file_name.data(), pixels, H, W, 1, 0, 8);
					dprintf(0, "\n\t> %s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\nLocal Measure Linear > %s saved.", file_name.data());
					}
					delete[] pixels;
				
				}
				
				//Copy to pixel buffer applying gamma correction and other stuff
				if(debug) dprintf(0, "\nCopy new data to pixels buffer applying back gamma correction... ");
				
				if(print_gamma)
				{
					dprintf(0, "Printing iteration %d", it);
					std::string file_name = out_name + ".png";
					//unsigned char* p = pixels;
					//unsigned char * pix = new unsigned char[H * W * nchannels];
					unsigned char *p = pixels;
					for(int i = 0; i< H*W; i++)
					{
						//Change first color channels
						for(int ch = 0; ch < nch_no_alpha; ch ++){
							type val = ((type)(host_temp[ch * H * W + i]));
							
							if(scale_back)
								val = val / scale;
							
							if(val <= 0.0031308)
								val = 12.92 * val;
							else
								val = 1.055 * pow( val, 1.0/2.4 ) - 0.055;
							*p = (unsigned char)(val * RGB_REF);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = nch_no_alpha; ch < nchannels; ch ++){
							p++;
						}
					}
					if(debug) dprintf(0, "done.\n");

					// Save PNG image
					dprintf(0, "\nOUTPUT PNG with GAMMA correction ...");
					save_image(file_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
					dprintf(0, "\n > %s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\n > %s saved.", file_name.data());
					}
					//delete[] pix;
				}

				
				// If we want to print linear space and default ouput was not in linear space
				if(print_linear)
				{
					std::string file_name = out_name + "-linear.png";
					//unsigned char *pix = new unsigned char[H * W * nchannels];
					unsigned char* p = pixels;
					for(int i = 0; i< H*W; i++)
					{
						//Change first color channels
						for(int ch = 0; ch < nch_no_alpha; ch ++){
							type val = ((type)(host_temp[ch * H * W + i]));
							
							if(scale_back)
								val = val / scale;
							*p = (unsigned char)(val * RGB_REF);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = nch_no_alpha; ch < nchannels; ch ++){
							p++;
						}
					}

					// Save PNG image
					dprintf(0, "\nOUTPUT PNG in LINEAR space ...");
					save_image(file_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
					dprintf(0, " \n > %s saved.\n", file_name.data());
					
					if(save_log)
					{
						fprintf(log_file, "\n%s saved.", file_name.data());
					}
					//delete[] pix;
				}
				
				// Print diffs with input to png
				if(print_diff_gamma)
				{
					std::string file_name = out_name + "-diff.png";
					//unsigned char* p = pixels;
					//unsigned char * pix = new unsigned char[H * W * nchannels];
					unsigned char *p = pixels;
					for(int i = 0; i< H*W; i++)
					{
						//Change first color channels
						for(int ch = 0; ch < nch_no_alpha; ch ++){
							type val = abs((host_temp[ch * H * W + i] - host_f[ch * H * W + i]));
							
							if(scale_back)
								val = val / scale;
							
							if(val <= 0.0031308)
								val = 12.92 * val;
							else
								val = 1.055 * pow( val, 1.0/2.4 ) - 0.055;

							*p = (unsigned char)(val * RGB_REF);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = nch_no_alpha; ch < nchannels; ch ++){
							p++;
						}
					}
					if(debug) dprintf(0, "done.\n");

					// Save PNG image
					dprintf(0, "\nPrinting DIFF PNG with GAMMA correction ... \n");
					save_image(file_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
					dprintf(0, "\n\t > %s saved.\n", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\n\t > %s saved.\n", file_name.data());
					}
					//delete[] pix;
				}

				
				// If we want to print linear space and default ouput was not in linear space
				if(print_diff_linear)
				{
					std::string file_name = out_name + "-diff_linear.png";
					//unsigned char *pix = new unsigned char[H * W * nchannels];
					unsigned char* p = pixels;
					for(int i = 0; i< H*W; i++)
					{
						//Change first color channels
						for(int ch = 0; ch < nch_no_alpha; ch ++){
							type val = abs(host_temp[ch * H * W + i] - host_f[ch * H * W + i]);
							
							if(scale_back)
								val = val / scale;
							*p = (unsigned char)(val * RGB_REF);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = nch_no_alpha; ch < nchannels; ch ++){
							p++;
						}
					}

					// Save PNG image
					dprintf(0, "Printing DIFF PNG in LINEAR SPACE ...\n");
					save_image(file_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
					dprintf(0, "\n\t > %s saved.\n", file_name.data());
					
					if(save_log)
					{
						fprintf(log_file, "%s saved.\n", file_name.data());
					}
					//delete[] pix;
				}	
				
				// Print diffs with input to png calculating rgb norm (single -channel image)
				if(print_diff_single_gamma)
				{
					std::string file_name = out_name + "-diff_single.png";
					//unsigned char* p = pixels;
					unsigned char * pixels = new unsigned char[H * W];
					unsigned char *p = pixels;
					
					for(int i = 0; i< H*W; i++)
					{
						type sum = 0;
						//Change first color channels
						for(int ch = 0; ch < nch_no_alpha; ch ++){
							type val = abs((host_temp[ch * H * W + i] - host_f[ch * H * W + i]));
							sum += val * val;	
						}
						sum = sqrt(sum / nch_no_alpha);
						if(scale_back)
								sum = sum / scale;
						
						// Gamma correction
						if(sum <= 0.0031308)
							sum = 12.92 * sum;
						else
							sum = 1.055 * pow( sum, 1.0/2.4 ) - 0.055;
						*p = (unsigned char)(sum * RGB_REF);
						p++;
						//Skip possible alpha channel (last channel)
					}
					if(debug) dprintf(0, "done.\n");

					// Save PNG image
					dprintf(0, "\nPrinting DIFF SINGLE image with GAMMA correction ...");
					save_image(file_name.data(), pixels, H, W, 1, 0, 8);
					dprintf(0, "\n\t > %s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\n\t > %s saved.\n", file_name.data());
					}
					delete[] pixels;
				}

				
				if(print_diff_single_linear)
				{
					std::string file_name = out_name + "-diff_single_linear.png";
					//unsigned char* p = pixels;
					unsigned char * pixels = new unsigned char[H * W];
					unsigned char *p = pixels;
					
					for(int i = 0; i< H*W; i++)
					{
						type sum = 0;
						//Change first color channels
						for(int ch = 0; ch < nch_no_alpha; ch ++){
							type val = abs((host_temp[ch * H * W + i] - host_f[ch * H * W + i]));	
							sum += val * val;	
						}
						sum = sqrt(sum / nch_no_alpha);
						if(scale_back)
								sum = sum / scale;
						*p = (unsigned char)(sum * RGB_REF);
						p++;
					}
					if(debug) dprintf(0, "done.\n");

					// Save PNG image
					dprintf(0, "\nPrinting DIFF SINGLE image in LINEAR space ...");
					save_image(file_name.data(), pixels, H, W, 1, 0, 8);
					dprintf(0, "\n%s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "%s saved.\n", file_name.data());
					}
					delete[] pixels;
				}	
				
				
				// Save txt version if wanted
				if (save_txt)
				{
					dprintf(0, "Saving TXT version in LINEAR space. It could generate a heavy file. Alpha channel will NOT be saved.");
					printToTxt((out_name + ".txt").data(), host_temp, H, W, nch_no_alpha);	
					dprintf(0, "\n%s saved.", (out_name + ".txt").data());
					if(save_log)
					{
						fprintf(log_file, "\n%s saved.\n", (out_name + ".txt").data());
					}
				}
				
				// For each index in the slice list, save a txt file with the data. In columns for gnuplot
				if(h_slice_list.size() > 0)
				{
					for(int index = 0; index < (int)h_slice_list.size(); index++)
					{
						int i = h_slice_list.at(index);
						std::string slice_file_name = out_name + "-hslice_" + Patch::to_string(i) + ".txt";
						FILE *slice_file = fopen(slice_file_name.data(), "w");
						 
						for(int j = 0; j < W; j++)
						{
							for(int ch = 0; ch < nch_no_alpha; ch ++)
							{
								fprintf(slice_file, "%.8f\t", host_temp[ch * W * H + i * W + j]);
							}
							fprintf(slice_file, "\n");
						}
						fclose(slice_file);
						dprintf(0, "\nH SLICE %d saved in %s.", i, (slice_file_name ).data());						
					}
				}
				
				if(v_slice_list.size() > 0)
				{
					for(int index = 0; index < (int)v_slice_list.size(); index++)
					{
						int j = v_slice_list.at(index);
						std::string slice_file_name = out_name + "-vslice_" + Patch::to_string(j) + ".txt";
						FILE *slice_file = fopen(slice_file_name.data(), "w");
						 
						for(int i = 0; i < W; i++)
						{
							for(int ch = 0; ch < nch_no_alpha; ch ++)
							{
								fprintf(slice_file, "%.8f\t", host_temp[ch * W * H + i * W + j]);
							}
							fprintf(slice_file, "\n");
						}
						fclose(slice_file);
						dprintf(0, "\nV SLICE %d saved in %s.", j, (slice_file_name ).data());						
					}
				}
				
				if(make_contrast_enhacement)
				{
					for(int i = 0; i < (int)ce_factor_list.size(); i++)
					{
						type ce_factor = ce_factor_list.at(i);
						
						std::string file_name = out_name + "-ce_" + Patch::to_string_f(ce_factor, 2) + ".png";
						unsigned char *p = pixels;
						for(int i = 0; i< H*W; i++)
						{
							//Change first color channels
							for(int ch = 0; ch < nch_no_alpha; ch ++){
								int ind = ch * H * W + i;
								type val = host_temp[ind] + ce_factor * ( host_f[ind] - host_temp[ind]) ;
								if(val < 0) val = 0.0;
								if(val > 1) val = 1.0;
								if(scale_back)
									val = val / scale;
								
								if(val <= 0.0031308)
									val = 12.92 * val;
								else
									val = 1.055 * pow( val, 1.0/2.4 ) - 0.055;
								*p = (unsigned char)(val * RGB_REF);
								p++;
							}
							//Skip possible alpha channel (last channel)
							for(int ch = nch_no_alpha; ch < nchannels; ch ++){
								p++;
							}
						}
						if(debug) dprintf(0, "done.\n");

						// Save PNG image
						dprintf(0, "\n Printing CONTRAST ENHANCEMENT PNG with GAMMA correction ...");
						save_image(file_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
						dprintf(0, "\n > %s saved.", file_name.data());
						if(save_log)
						{
							fprintf(log_file, "\n > %s saved.", file_name.data());
						}
					}

				}
				
				if(make_contrast_enhacement_linear)
				{
					for(int i = 0; i < (int)ce_factor_list.size(); i++)
					{
						type ce_factor = ce_factor_list.at(i);
						
						std::string file_name = out_name + "-ce-linear-" + Patch::to_string_f(ce_factor, 2) + ".png";
						unsigned char *p = pixels;
						for(int i = 0; i< H*W; i++)
						{
							//Change first color channels
							for(int ch = 0; ch < nch_no_alpha; ch ++){
								int ind = ch * H * W + i;
								type val = host_temp[ind] + ce_factor * ( host_f[ind] - host_temp[ind]) ;
								
								if(scale_back)
									val = val / scale;
								*p = (unsigned char)(val * RGB_REF);
								p++;
							}
							//Skip possible alpha channel (last channel)
							for(int ch = nch_no_alpha; ch < nchannels; ch ++){
								p++;
							}
						}
						if(debug) dprintf(0, "done.\n");

						// Save PNG image
						dprintf(0, "\n Printing CONTRAST ENHANCEMENT PNG in LINEAR space ...");
						save_image(file_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
						dprintf(0, "\n > %s saved.", file_name.data());
						if(save_log)
						{
							fprintf(log_file, "\n > %s saved.", file_name.data());
						}
					}
				}				

				// Back to false stage
				print_it = false;
	
			}	
			
			// Check if we still want to run iterations
			run = (calc_convergence && (!converged)) || (it < nit) ;	// If there is print left go on. If we have not reached convergence also.
			run = run && ( max_it < 0 || (it < max_it) );	// If it >= max_it then we dont want to go on provided max_it >= 0
			if (converged && force_stop_on_convergence) run = false;	// if force-stop we dont want to go on after convergence
		}

	}
	}
	}
	}
	
	printf("\n");

	
	if(debug) dprintf(0, "Cleaning memory in host ... ");
	delete[] host_f;
	delete[] host_g;
	delete[] pixels;
	delete[] gpixels;
	delete[] host_temp;
	delete[] host_local_measure;
	if(debug) dprintf(0, "done.\n");
	
	if(debug) dprintf(0, "Cleaning memory in device ... ");
	cudaFree(dev_f);
	cudaFree(dev_g);
	cudaFree(dev_g_last);
	cudaFree(dev_stdev);
	cudaFree(dev_temp);
	cudaFree(dev_temp2);
	if(debug) dprintf(0, "done\n");
	
	if(save_log)
		fclose(log_file);
	
	return 1;
	

}



// Function to print png, still not usable
void print_png(type *host_g, std::string out_name, int H, int W, int nchannels_no_alpha, int nchannels,  int color_type, int bit_depth, bool gammacor, type scale)
{
	unsigned char *pixels = new unsigned char[H * W * nchannels_no_alpha];
	//Copy to pixel buffer applying gamma correction and other stuff
	unsigned char* p = pixels;
	for(int i = 0; i< H*W; i++)
	{
		//Change first color channels
		for(int ch = 0; ch < nchannels_no_alpha; ch ++){
			type val = ((type)(host_g[ch * H * W + i]));
			
			val = val / scale;
			
			if(gammacor){
				if(val <= 0.0031308)
					val = 12.92 * val;
				else
					val = 1.055 * pow( val, 1.0/2.4 ) - 0.055;
			}
			*p = (unsigned char)(val * (pow(2, bit_depth) -1));
			p++;
		}
		//Skip possible alpha channel (last channel)
		for(int ch = nchannels_no_alpha; ch < nchannels; ch ++){
			p++;
		}
	}

	// Save PNG image
	save_image(out_name.data(), pixels, H, W, nchannels, color_type, bit_depth);
	printf("%s saved.\n", out_name.data());
	free(pixels);
}


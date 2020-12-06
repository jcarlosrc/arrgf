// Cuda implementation of the RRGF filter
#include<algorithm>
#include <stdio.h>
#include <math.h>
#include <chrono>

#include "png_tool.h"
#include <string>
#include <vector>
#include <ctime>

// RGBE support from https://www.graphics.cornell.edu/~bjw/rgbe.html
/*extern "C"
{
	#include "rgbe.h"
}*/
// HDR loading support from Igor Kravtchenko github
#include "hdrloader.h" 

#include "convolution_kernel.h"
#include "common_math_kernel.h"
#include "patch.h"
#include "local_measure.h"
#include "trilateral_kernel.h"
#include "rgbnorm2_kernel.h"
#include "l2loc2_kernel.h"
#include "matrix.h"
#include "util.h"

// Include color space conversion
//#include "ColorSpace-master/src/ColorSpace.h"
//#include "ColorSpace-master/src/Comparison.h"
//#include "ColorSpace-master/src/Conversion.h"
//#include "ColorSpace-master/src/Conversion.h"


//Data type to use for images and images in kernels
#define type float
//typename type = float;

#ifndef CTE255
#define CTE255 255.0f
#endif

#define IMAGE_PNG 0
#define IMAGE_HDR 1
#define IMAGE_TXT 2

std::string get_im_format(int format)
{
	switch(format)
	{
		case 0:
		return ".png";
		break;

		case 1:
		return ".hdr";
		break;

		default:
		return ".";
	}
}

// kernels
#define KER_NONE -1
#define KER_GAUSSIAN 0
#define KER_TUKEY 1
#define KER_BOX 2
#define KER_LORENTZ 3
#define KER_HSINC 4
#define KER_SINC 5
#define KER_DELTA 6

std::string get_ker_name(int ker)
{
	switch(ker){
		case KER_NONE:
			return "none";
			break;
		case KER_GAUSSIAN:
			return std::string("gaussian");
			break;
		case KER_TUKEY:
			return std::string("tukey");
			break;
		case KER_BOX:
			return std::string("box");
			break;
		case KER_LORENTZ:
			return std::string("lorentz");
			break;
		case KER_HSINC:
			return std::string("hsinc") ;
			break;
		case KER_SINC:
			return std::string("sinc") ;
			break;
		case KER_DELTA:
			return std::string("delta");
			break;
		default:
			return std::string("none");
	}
}

// Local measure types
#define LM_NONE -1
#define LM_MAX 0
#define LM_MEAN 1
#define LM_STDEV 2

std::string get_lm_name(int lm)
{
	switch(lm)
	{
		case LM_NONE:
			return "none";
			break;
		case LM_MAX:
			return "max";
			break;
		case LM_MEAN:
			return "mean";
			break;
		case LM_STDEV:
			return "stdev";
			break;
		default:
			return "none";
	}
}

#define W_BOX 0
#define W_CIRCLE 1
#define W_GAUSSIAN 2

std::string get_w_name(int w)
{
	switch(w)
	{
		case W_BOX:
			return "box";
			break;
		case W_CIRCLE:
			return "circle";
			break;
		case W_GAUSSIAN:
			return "gaussian";
			break;
		default:
			return "box";
	}
}

// Algorithms
#define CONF_CONV 4
#define CONF_BIL 3
#define CONF_RGF 2
#define CONF_RRGF 1
#define CONF_ARRGF 0
#define CONF_NONE -1
#define CONF_GAMMIFY 5
#define CONF_UNGAMMIFY 6
#define CONF_TONE_MAPPING 7
#define CONF_TO_GRAY 8

std::string get_conf_name(int conf)
{
	switch(conf)
	{
		case CONF_NONE:
			return "none";
			break;
		case CONF_ARRGF:
			return "ARRGF";
			break;
		case CONF_RRGF:
			return "RRGF";
			break;
		case CONF_RGF:
			return "RGF";
			break;
		case CONF_BIL:
			return "Bilateral";
			break;
		case CONF_CONV:
			return "Convolution";
			break;
		case CONF_GAMMIFY:
			return "GAMMA";
			break;
		case CONF_UNGAMMIFY:
			return "LINEAR";
			break;
		case CONF_TONE_MAPPING:
			return "TM";
			break;
		case CONF_TO_GRAY:
			return "GRAY";
			break;
		default:
			return "none";
	}
}

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

// ------------------------------------------------------------------------------------------------------
/*----------------------  VARIABLES ---------------------------------------------------------------------*/
// -------------------------------------------------------------------------------------------------------

int input_format = IMAGE_PNG;	// png image
int output_format = IMAGE_PNG;

bool print_alpha_channel = false;
bool print_input_png = false;

int H, W;	// Input widht and height
int domain_extension = 0;
bool domain_extension_exp = false;

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
type calibration_xicxss = 0.4375;
bool xic_exp = false;


std::vector<std::string> conf_all;
std::vector<std::string> conf_list;

std::vector<type> scaleValues;
int scale_exp = 0;
//type scale = 1.0f;
int scale_back = false;
std::string scale_back_name = std::string("no");
type scale;

// Spatial kernel specs
std::vector<type> ssValues;
int ss_exp = 0;
int ss_mod = 1;
type ss;

type gaussian_neigh = 4.0;

int sker_mod = 1;	// allow modifications?
int sker_exp = 0;	// is explicitly defined?
int sker = 0; // gaussian exp(-0.5x**2/ss**2) spatial kernel by default
//type ss =  0.355 / cutoff;	// sigma value for gaussian
int sskh = 0;
int sskw = 0;

std::vector<type> srValues;
int sr_exp = 0;
int sr_mod = 1;
type sr;

int rker_mod = 1;
int rker_exp = 0;
int rker = 0; // gaussian intensity kernel by default	

int regker_mod = 1;
int regker_exp = 0;
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
int it;

bool print_gamma = false;
bool gammacor_in = true;
bool print_linear = false;
bool gammacor_load_it = true;
bool gammacor_txt_out = false;	// Txt is saved in linear space. ALWAYS.

bool print_input_txt = false;

int adaptive_mod = 1;
std::string adaptive_name = "yes";
int adaptive_exp = 0;
int adaptive = 1;

// Local measure for adaptativeness
int local_measure = 0; // max
type sr_ref = 0.48;
int local_measure_mod = 1;
int local_measure_exp = 0;

// Local measure weights 
int lweights = 2;	// Gaussian, to achieve smoother transitions. Specially important for lm = ma
int lweights_mod = 1;
bool lweights_exp = false;
int lkh = 0;
int lkw = 0;

type sl;
int sl_mod = 1;
int sl_exp = 0;

// Regularization convolution for local measure (like gaussian blur to smooth results)
int lmreg_ker = KER_NONE;
type lmreg_s = 0.0f;
bool lmreg_s_exp = false;
int lmreg_ker_mod = 1;
int lmreg_ker_exp = 0;
int lmreg_kh;
int lmreg_kw;

int mker_mod = 1;
int mker_exp = 0;
int mker = -1; // No median kernel by default
type sm = 0;
int sm_mod = 1;

int conf_mod = 1;
int conf_exp = 0;
int conf = 0; // arrgf by default

// M = support size for sinc and hsinc approximation
type M = 20;

type infsr = 1.0e-5;
bool infsr_exp = false;

int i0 = 10;
int iend = 10;
int print_exp = 0;
int max_it = -1;
bool print_all = false;

/* CONVERGENCE */

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

/* OTHER OUTPUT */
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

bool make_hdr_tone_mapping = false;


// ** Adapt sr value for RGF as in thesis
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

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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
int set_conf ()
{
	switch(conf)
	{
		case CONF_BIL:
		adaptive = 0; adaptive_mod = 0;
		regker = KER_NONE; sreg = 0; regker_mod = 0;
		mker = KER_NONE; sm = 0; mker_mod = 0;
		local_measure = -1; sl = 0; local_measure_mod = 0; sl_mod = 0;
		if(nit_mod && !nit_exp) nit = 1;
		
		conf_mod = 0;
		conf_exp = 1;

		break;
	case CONF_RGF:
		adaptive = 0; adaptive_mod = 0;
		regker = -1; sreg = 0; regker_mod = 0;
		mker = KER_NONE; sm = 0; mker_mod = 0;
		local_measure = -1; sl = 0;  local_measure_mod = 0; sl_mod = 0;
		if(nit_mod && !nit_exp) nit = 10;
		conf_mod = 0;
		conf_exp = 1;
		break;
	case CONF_RRGF:
		adaptive = 0; adaptive_mod = 0;
		mker = KER_NONE; sm = 0;  mker_mod = 0;
		local_measure = -1; sl = 0; local_measure_mod = 0;
		if(nit_mod && !nit_exp) nit = 10;
		
		conf_mod = 0;
		conf_exp = 1;
		break;
	
	case CONF_ARRGF:
		adaptive = 1; adaptive_mod = 0;
		mker = KER_NONE; sm = 0; mker_mod = 0;
		if(nit_mod && !nit_exp) nit = 10;
		
		conf_mod = 0;
		conf_exp = 1;

		return conf;
		break;
	case CONF_CONV:
		adaptive = 0; adaptive_mod = 0;
		regker = KER_NONE; sreg = 0; regker_mod = 0;
		mker = KER_NONE; sm = 0; mker_mod = 0;
		local_measure = LM_NONE; sl = 0;  local_measure_mod = 0;
		rker = -1; rker_mod = 0;
		if(nit_mod && !nit_exp) nit = 1;
		
		conf_mod = 0;
		conf_exp = 1;
		break;
	default:
		break;
	}
	return conf;
}

void make_output_name(std::string *out_name)
{
	// Add adapt_sr info to RGF
	if(adapt_sr && adaptive == false)
	{
		*out_name += "_asr";
	}

	if(rker_exp)
	{
		*out_name += "-rker-" + get_ker_name(rker);
	}
	if(sr > 0 )
	{
		*out_name += "-sr-" +  Patch::to_string_f(sr, 2);
	}
	if(sker_exp)
	{
		*out_name += "-sker-" + get_ker_name(sker);
	}
	if(xic_exp == false)
	{
		*out_name += "-ss-" +  Patch::to_string_f(ss, 2);
	}
	else
	{
		*out_name += "xic_1_" + Patch::to_string_f(ss / calibration_xicxss,2);
	}
	if(sreg_exp) *out_name += "-sreg_" + Patch::to_string_f(sreg, 2);
	// Add Adaptive Filter info
	if (adaptive && ss > 0 && local_measure_exp)
	{
		*out_name += "-lm_" + get_lm_name(local_measure);
		*out_name += "_" + get_w_name(lweights) + "_" + Patch::to_string_f(sl, 2);
	
		if(lmreg_ker_exp)
		{
			*out_name += "-lmreg_" + get_ker_name(lmreg_ker) + "_" + Patch::to_string_f(lmreg_s, 2);
		}
	}
	
	// Add iteration info
	*out_name += "-nit-" + Patch::to_string(it);

	// Add scaling info
	if (scale_exp)
	{
		*out_name += "-scale" ;
		if (scale_back)
		{
			*out_name += "_sb";
		}
		*out_name += "-" + Patch::to_string_f(scale, 2);
	}
	
	// Add infsr info
	if(infsr_exp == true)
	{
		*out_name += "-infsr_" + Patch::to_string_f(infsr, 2);
	}

}

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

	// Read PARAMETERS
	int argi = 1;
	while(argi < nargs)
	{	
	
		std::string pname = std::string(args[argi]);
		argi ++;

		/* GENERAL CONFIGURATIONS */
		if(pname == "-gauss-neigh" || pname == "-gn" || pname == "-gauss")
		{
			gaussian_neigh = atof(args[argi]);
			argi ++;
		}
		if(pname == "-ext-zeros" || pname == "-zeros")
		{
			domain_extension = 0;
			domain_extension_exp = true;
		}
		if(pname == "-ext-mirror" || pname == "-mirror")
		{
			domain_extension = 1;
			domain_extension_exp = true;
		}

		/* INPUT AND OUTPUT TYPES AND SPECS */

		if(pname == "-input" || pname == "-in")
		{
			//printf("Input : ");
			input_name = std::string(args[argi]);
			argi++;
			//printf("%s\n", input_name.data());
		}

		if(pname == "-output"|| pname == "-out")
		{
			//printf("Output: ");
			output_name = std::string(args[argi]);
			argi++;
			//printf("%s\n", output_name.data());
			make_output_default = 0;
			
		}
		if(pname == "-hdr-input" || pname == "-input-hdr" || pname == "-hdr")
		{
			input_format = IMAGE_HDR;
		}

		if(pname == "-hdr-tone-mapping" || pname == "-tone-mapping" || pname == "-print-tone-mapping" || pname == "-make-tone-mapping")
		{
			input_format = IMAGE_HDR;
			gammacor_in = false;			
			make_hdr_tone_mapping = true;
		}

		/* ------- OTHER OUTPUT FILES */

		if (pname == "-print-input-txt" || pname == "-print-txt-input")
		{
			print_input_txt = true;
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
		
		if (pname.compare("-log-auto") == 0 || pname.compare("-make-log") == 0)
		{
			save_log = true;
			log_file_name = std::string("");
			auto_log_string = true;
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
			argi = Util::read_input_list(slice_list, args, argi, nargs);

		}
		if(pname == "-slice-only" || pname == "-only-slices" || pname == "print-slice-only")
		{
			print_gamma = false;
			print_linear = false;
		}

		if(pname.compare("-txt") == 0 || pname.compare("-save-txt") == 0 || pname == "-print-txt")
		{
			save_txt = true;
		}
		if(pname == "-txt-only" || pname == "-save-txt-only")
		{
			save_txt = true;
			print_gamma = false;
			print_linear = false;
		}

		// Gamma or linear input/output
		/*if (pname.compare("-g") == 0|| pname.compare("-gamma") == 0 )
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
		}*/
		if(pname.compare("-in-linear") == 0 || pname.compare("-linear-in") == 0 || pname.compare("-linear-input") == 0 || pname.compare("-input-linear") == 0 || pname == "-lin-in" || pname == "-in-lin")
		{
			gammacor_in = false;
		}

		if(pname.compare("-out-linear-also") == 0 || pname.compare("-linear-also") == 0 || pname == "-linear-output-also" || pname == "-output-linear-also")
		{
			print_linear = true;
		}
		if(pname == "-print-linear")
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
		if(pname == "-print-gamma")
		{
			print_gamma = true;
		}


		// Print differences with input to PNG
		if(pname == "-print-diff-rgb" || pname == "-print-diff-rgb-gamma" || pname == "-print-diff-gamma" || pname == "-print-diff")
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
		if(pname == "-print-lm-gamma" || pname == "-print-local-measure-gamma" || pname == "-print-lm")
		{
			print_local_measure_gamma = true;
		}
		if(pname == "-print-lm-linear" || pname == "-print-local-measure-linear")
		{
			print_local_measure_linear = true;
		}
		
		// Contrast enhancement
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
			argi = Util::read_input_list<type>(&ce_factor_list, args, argi, nargs);
		}
		if(pname == "-ce-linear" || pname == "-contrast-enhancement-linear")
		{
			make_contrast_enhacement_linear = true;
			ce_factor_list.clear();
			argi = Util::read_input_list(&ce_factor_list, args, argi, nargs);
		}

		/* ALGORITHMS */

		if(pname.compare("-arrgf") == 0)
		{
			if(conf_mod)
			{
				conf = CONF_ARRGF;
				set_conf();
			}
		}
		if(pname.compare("-rrgf") == 0)
		{
			if(conf_mod)
			{
				conf = CONF_RRGF;
				set_conf();
			}
		}
		if(pname.compare("-rgf") == 0)
		{
			if(conf_mod)
			{
				conf = CONF_RGF;
				set_conf();
			}
		}
		if(pname.compare("-conv") == 0 || pname.compare("-convolution") == 0)
		{
			if(conf_mod)
			{
				conf = CONF_CONV;
				set_conf();
			}
		}
		if(pname.compare("-bilateral") == 0|| pname.compare("-bil") == 0)
		{
			if(conf_mod)
			{
				conf = CONF_BIL;
				set_conf();
			}
		}

		if(pname == "-gamma" || pname == "-gammify")
		{
			gammacor_in = false;
			conf = CONF_GAMMIFY;
			set_conf();
		}
		if(pname == "-ungamma" || pname == "-ungammify")
		{
			gammacor_in = true;
			conf = CONF_UNGAMMIFY;
			set_conf();
		}

		if(pname == "-to-gray" || pname == "-gray")
		{
			conf = CONF_TO_GRAY;
			set_conf();
		}

		/* CONVERGENCE */
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
			else
			{
				argi --;
				argi = Util::read_and_validate_input_list(&show_norm_list, &all_norms, args, argi, nargs, false);
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
			else
			{
				argi --;
				argi = Util::read_and_validate_input_list(&conv_norm_list, &all_norms, args, argi, nargs, false);
			}
		}
		
		// Check for list of eps for convergence
		if(pname.compare("-conv-eps") == 0)
		{
			conv_eps_list.clear();
			argi = Util::read_input_list(&conv_eps_list, args, argi, nargs);
			conv_eps_exp = true;
			calc_norms = true;
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

		// add ALL norms to the convergence criteria
		if (pname.compare("-conv-norm-all") == 0 || pname.compare("-conv-all-norms") == 0)
		{			
			
			conv_norm_exp = true;
			check_and_add(&conv_norm_list, all_norms);
			calc_norms = true;

		}

		/* ITERATIONS */	
		
		if(pname.compare("-max-it") == 0)
		{
			max_it = atoi(args[argi]);
			printf("-max-it");
			argi++;
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
				argi = Util::read_input_list(&it_list, args, argi, nargs);
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

		/* ARRGF PARAMETERS */
		
		// Check for list of sr values
		if(pname.compare("-sr") == 0)
		{
			srValues.clear();
			argi = Util::read_input_list(&srValues, args, argi, nargs);
			sr_exp = true;
		}
		if(pname == "-fixed-sr" || pname == "-non-adaptive-sr")
		{
			adapt_sr = false;
		}
		
		// Check for list of ss values
		if(pname.compare("-ss") == 0)
		{
			ssValues.clear();
			argi = Util::read_input_list(&ssValues, args, argi, nargs);			
		}
		// Regularization
		if(pname.compare(std::string("-sreg")) == 0 || pname.compare("-s") == 0 || pname.compare("-reg-sigma") == 0 || pname.compare("-regularization-sigma") == 0 || pname.compare("-regs") == 0)
		{
			if(sreg_mod )
			{
				sreg_exp = true;
				sregValues.clear();
				argi = Util::read_input_list(&sregValues, args, argi, nargs);
			}
		}
		if (pname == "-xic")
		{
			ssValues.clear();
			argi = Util::read_input_list(&ssValues, args, argi, nargs);
			xic2ss<type>(ssValues, calibration_xicxss);
			xic_exp = true;
		}
		
		// Change xicxss
		if (pname == "-xicxss" || pname == "-calibration-xicxss")
		{
			calibration_xicxss = atof(args[argi]);
			argi++;
		}

		/* INPUT SCALING */

		if(pname == "-scale")
		{
			scaleValues.clear();
			argi = Util::read_input_list(&scaleValues, args, argi, nargs);
			scale_exp = true;

		}
		
		if (pname == "-sb" || pname == "-scale-back")
		{
			scale_back = 1;
			scale_back_name = "yes";
		}
		
		// Show help
		if(pname == "-help")
		{
			showHelp();	
			argi++;
		}
		
		if(pname == "-infsr")
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
		if(pname == "-cutoff")
		{
			cutoff = atof(args[argi]);
			argi++;
		}
		if(pname == "-lm" || pname == "-local-measure" || pname == "-local_measure" || pname == "-lmker" || pname == "local-measure-kernel")
		{
			std::string temp = std::string(args[argi]);
			argi ++;

			if(temp.compare(std::string("max")) == 0)
			{
				local_measure = LM_MAX;
				local_measure_exp = 1;
				sr_ref = 0.48;
			}
			if(temp.compare(std::string("mean")) == 0)
			{
				local_measure = LM_MEAN;
				local_measure_exp = 1;
				sr_ref = 0.33;
			}
			if(temp.compare(std::string("stdev")) == 0)
			{
				local_measure = LM_STDEV;
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
					lweights = W_CIRCLE;
					lweights_exp = true;
				}
				if(option == "gaussian")
				{
					lweights = W_GAUSSIAN;
					lweights_exp = true;
				}
				if(option == "none" || option == "box" || option == "constant")
				{
					lweights_exp = true;
					lweights = W_BOX;
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
				lmreg_ker = KER_NONE;
				lmreg_s = 0;
				lmreg_ker_exp = 1;
			}else
			if(temp.compare(std::string("gaussian")) == 0){
				lmreg_ker = KER_GAUSSIAN;
				lmreg_ker_exp = 1;
				
			} else
			if(temp.compare(std::string("tukey")) == 0) {
				lmreg_ker = KER_TUKEY;
				lmreg_ker_exp = 1;
			} else
			if(temp.compare(std::string("box")) == 0) {
				lmreg_ker = KER_BOX;
				lmreg_ker_exp = 1;
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
				lmreg_ker = KER_GAUSSIAN;
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
				
				sker = KER_GAUSSIAN;
				sker_exp = 1;
				
			} else
			if(temp.compare(std::string("tukey")) == 0) {	
				sker_exp = 1;
				sker = KER_TUKEY;		
				
			} else
			if(temp.compare(std::string("box")) == 0) {	
				sker_exp = 1;
				sker = KER_BOX;
			} else
			if(temp.compare(std::string("lorentz")) == 0) {
				sker_exp = 1;
				sker = KER_LORENTZ;

			}
			else if(temp == "hamming-sinc" || temp == "hsinc")
			{
				sker_exp =1;
				sker = KER_HSINC;
				M = atof(args[argi]);
				argi --;

			}
			else if(temp.compare(std::string("sinc")) == 0)
			{
				sker_exp = 1;
				sker = KER_SINC;
				M = atof(args[argi]);
				argi --;
				
			}
			else if(temp.compare("delta") == 0)
			{
				sker_exp = 1;
				sker = KER_DELTA;
			}
			else{
				argi --;
			}
			
		}

		// Range kernel
		if(rker_mod && ( pname == "-rker" || pname == "-intensity-kernel" || pname == "-range-kernel" || pname == "-range" ))
		{	
			
			std::string temp = std::string(args[argi]);
			argi ++;
			
			if(temp.compare(std::string("none")) == 0 || temp.compare(std::string("off")) == 0){	
				rker = KER_NONE;
				
				srValues.clear();
				srValues.push_back(0.0f);
				
				rker_exp = 1;
			} else
	
			if(temp.compare(std::string("gaussian")) == 0){	
				rker = KER_GAUSSIAN; 			
				rker_exp = 1;
			} else
			if(temp.compare(std::string("tukey")) == 0) {	
				rker = KER_TUKEY;			
				rker_exp = 1;
			} else
			if(temp.compare(std::string("box")) == 0) {		
				rker = KER_BOX;
				rker_exp = 1;
			} else 
			if(temp.compare(std::string("lorentz")) == 0) {
				rker = KER_LORENTZ;
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
				regker = KER_NONE;
				sreg = 0;
				regker_exp = 1;
			}else
	
			if(temp.compare(std::string("gaussian")) == 0){
				regker = KER_GAUSSIAN;
				regker_exp = 1;
			} else
			if(temp.compare("tukey") == 0)
			{
				regker = KER_TUKEY;
				regker_exp = 1;
			} else
			if(temp.compare(std::string("box")) == 0) {
				regker = KER_BOX;
				regker_exp = 1;
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
				mker = KER_NONE;
				sm = 0;				
				mker_exp = 1;
			}
			if(temp.compare(std::string("gaussian")) == 0)
			{
				mker = KER_GAUSSIAN;
				mker_exp = 1;
			} else
			if(temp.compare(std::string("tukey")) == 0)
			{
				mker = KER_TUKEY;				
				mker_exp = 1;
			} else
			if(temp.compare(std::string("box")) == 0)
			{
				mker = KER_BOX;				
				mker_exp = 1;
			} else
			if(temp.compare(std::string("lorentz")) == 0)
			{
				mker = KER_LORENTZ;
				
				mker_exp = 1;
			} else
			{
				argi--;
			}
		}
 
	}

	// ************************ 	VALIDATION AND FILLING OF MISSING DATA, INITIALIZATION ALSO ************************************	
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

	
	// ******************* ARRGF PARAMETERS ********************************************************************8
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
		log_file_name = output_name + std::string(" - ") + get_conf_name(conf) + std::string(" - LOG - ") + std::string(dt) + std::string(".md");
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

	printf("\n\nALGORITHM:\t%s\n", get_conf_name(conf).data());
	if (save_log) fprintf(log_file, "\nALGORITHM CONFIGURATION:\t%s (%i)\n", get_conf_name(conf).data(), conf);
	
	printf("\nPARAMETERS:\n");
	
	printf("\nInput: \t%s\n", (input_name + get_im_format(input_format)).data());
	if(gammacor_in)
		printf("INPUT is GAMMA corrected.\n");
	else
		printf("INPUT is already in LINEAR SPACE.\n");
	
	if(save_log)
	{
		fprintf(log_file, "\nInput:\t%s\n", input_name.data());
	}
	
	//printf("Input scaling factor: %.2f\n", scale);
	printf("Output image prefix: %s\n", output_name.data());
	if (save_log) fprintf(log_file, "Output image: %s\n", output_name.data());
	
	if(conf == CONF_RGF || conf == CONF_ARRGF || conf == CONF_CONV)
	{
		printf("\nSpatial kernel:\t %s", get_ker_name(sker).data());
		if(save_log) fprintf(log_file, "\nSpatial kernel:\t %s", get_ker_name(sker).data());
		
		printf("\n\tss values: \t"); printfVector(ssValues); printf("\n");
		if(save_log)
		{
			fprintf(log_file, "\nss values: \t");
			fprintfVector(log_file, ssValues);
			fprintf(log_file, "\n");
		}

		printf("\nRegularization kernel:\t %s\n", get_ker_name(regker).data());
		if(save_log) fprintf(log_file, "Regularization kernel:\t %s", get_ker_name(regker).data());

		printf("\tsreg values: \t"); printfVector(sregValues); printf("\n");
		if(save_log)
		{
			fprintf(log_file, "\nsreg values: \t");
			fprintfVector(log_file, sregValues);
			fprintf(log_file, "\n");
		}
		
		printf("\nRange kernel:\t %s", get_ker_name(rker).data());
		if (save_log) fprintf(log_file, "Range kernel:\t %s", get_ker_name(rker).data());
		
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
			printf("\tLocal Measure:\t %s", get_lm_name(local_measure).data());
			if (save_log) fprintf(log_file, "\tLocal measure calculation kernel for adaptiveness:\t %s\n", get_lm_name(local_measure).data());
			if(!sl_exp)
				printf("\n\tLocal measure weights:\t%s, sl = auto\n", get_w_name(lweights).data());
			else
				printf("\n\tLocal measure weights: \t%s, sl = %f", get_w_name(lweights).data(), sl);
			if (save_log) fprintf(log_file, "\tLocal measure weights:\t%s, sl = %.4f\n", get_w_name(lweights).data(), sl);
			printf("\tLocal measure regularization: \t%s, lmreg-s = %.4f\n", get_ker_name(lmreg_ker).data(), lmreg_s);
			if (save_log) fprintf(log_file, "\tLocal measure regularization: \t%s, lmreg-s = %.4f\n", get_ker_name(lmreg_ker).data(), lmreg_s);
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
	
	}
	
	// ---------------- INPUT LOADING ---------------------------------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
		
	if(debug) printf("\n LOADING INPUT AND INITIALIZATION OF CPU / GPU ARRAYS .. \n");
	
	int f_nchannels, f_valid_channels;
	int gH, gW, gnchannels, gvalid_channels;
	unsigned char f_bit_depth;
	unsigned char gbit_depth;
	//pixel buffer
	unsigned char *pixels = NULL;
	unsigned char *gpixels = NULL;

	// Info for input of RGF algorithm. Can be gray version of f
	int nchannels, valid_channels;
	
	dprintf(0,"Reading input image f ... ");
	//read_png(args[1], H, W, nchannels, pixels, bit_depth, valid_channels);

	// Array to store input data
	type *host_input_f = NULL;
	int f_rgb_format = MATRIX_RGB_TYPE_INTER;

	int RGB_REF = pow(2, 8) -1;

	// ---------- READ INPUT using type defined by user ------------------------------------------------------------
	
	// HDR image reading. Output = RGB RGB RGB
	if(input_format == IMAGE_HDR)
	{
		dprintf(0, "\nReading IMAGE_HDR\n");

		HDRLoaderResult <type> *hdrData = new HDRLoaderResult<type>();
		HDRLoader *hdrLoader = new HDRLoader();

		const char* input = (input_name + get_im_format(input_format)).data();
		
		if( ! hdrLoader->load<type>(input , *hdrData ) ) {
			printf( "error loading %s \n", input );
			return 1;
		}
		// Assign H and W values and get data in format grayscale L_R L_R .. L_G L_G ...
		W = hdrData -> width;
		H = hdrData -> height;
		host_input_f = hdrData -> cols;

		// HDR reader reads in cols in format rgb LLL L = 	RGB RGB
		f_nchannels = 3;
		f_valid_channels = 3;
		f_rgb_format = MATRIX_RGB_TYPE_INTER;

		// Change format to RRRR GGGG BBBB
		type *temp = new type[H * W * f_nchannels];
		f_rgb_format = Matrix::change_rgb_format(host_input_f, temp, H * W, 3, f_rgb_format);
		std::swap(host_input_f, temp);
		delete[] temp;

		dprintf(0, "RGB format changed.\n");
		
		gammacor_in = false;
		load_it = false;

		dprintf(0, "Reading input HDR done.\n");
	
	}
	// Reading typical GRAY or RGB 8b images
	else if(input_format == IMAGE_PNG)
	{
		dprintf(0, "\nReading IMAGE_PNG image\n");
		read_png((input_name + get_im_format(input_format)).data(),  pixels, H, W, f_nchannels, f_bit_depth, debug);
		
		f_valid_channels = f_nchannels;
		if(f_nchannels == 2 || f_nchannels == 4) f_valid_channels -= 1;
		
		if(f_bit_depth != 8)
		{
			dprintf(0, "Reading Input: bit depth of %i is not supported yet. Aborting ... \n", f_bit_depth);
			return 0;
		}
		printf("Image width: %i, Image height: %i, Number of channels: %i, valid channels: %i\n",H, W, f_nchannels, f_valid_channels);

		// Initialize host_f
		host_input_f = new type[f_valid_channels * H * W];
		//HOST: Create arrays for each channel, alpha channel will not be filtered
		// Will save host iterations as well as norms to show

		for(int ch = 0; ch< f_valid_channels ; ch++){
			unsigned char *p = (pixels + ch);
			//fill channel i for f
			for(int i = 0; i< H*W; i++){
			
				type val = ((type)(*p))/RGB_REF;
				// Initialize images in Host
				host_input_f[ch * W * H + i] = val;			
				p += f_nchannels;
			}
		}

		f_rgb_format = MATRIX_RGB_TYPE_CHUNK;

		// Ungamma input to have linear input
		if(gammacor_in)
		{
			Matrix::ungamma<type>(host_input_f, host_input_f, H * W * f_valid_channels);
		}

		valid_channels = f_valid_channels;
		nchannels = f_nchannels;

		if(load_it)
		{
			read_png(it_from_image_name + ".png", gpixels, gH, gW, gnchannels,  gbit_depth, debug);
			if(gbit_depth != 8)
			{
				dprintf(0, "Reading Input Iteration: bit depth of %i is not supported yet. Aborting ... \n", gbit_depth);
				return 0;
			}
				// Images have to match
			if(gH != H || gW != W || gnchannels != f_nchannels)
			{
				dprintf(0, "Input and Iteration images do not match.\n");
				return 0;
			}
			
			gvalid_channels = gnchannels;
			if(gnchannels == 2 || gnchannels == 4) gvalid_channels -= 1; 
		}

	}

	// Max and min values of f
	dprintf(0, "Calculating min(f) and max(f) values\n");
	type f_max = Matrix::max_rgb_norm2<type>(host_input_f, H, W, f_valid_channels);
	type f_min = Matrix::min_rgb_norm2<type>(host_input_f, H, W, f_valid_channels);
	type f_range = sqrt(f_max) - sqrt(f_min);
	dprintf(0,  "\tmin(f) = %f and max(f) = %f \n", f_min, f_max);

	// ---------- Save txt slices for original function f also for comparisons ---------------------------------

	if(print_input_txt)
	{
		dprintf(0, "\nSaving TXT version for INPUT ...");
		std::string out_name = output_name;
		Util::printToTxt<type>(out_name , host_input_f, H, W, f_valid_channels, "%.8f\t", f_rgb_format);
	}
	
	if(h_slice_list.size() > 0)
	{
		printf("Saving H slices for input image f ... \n");
		for(int index = 0; index < (int)h_slice_list.size(); index++)
		{
			int i = h_slice_list.at(index);
			std::string slice_file_name = output_name + "-hslice_" + Patch::to_string(i) + ".txt";
			FILE *slice_file = fopen(slice_file_name.data(), "w");
			 
			for(int j = 0; j < W; j++)
			{
				for(int ch = 0; ch < f_valid_channels; ch ++)
				{
					fprintf(slice_file, "%.8f\t", host_input_f[ch * W * H + i * W + j]);
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
			std::string slice_file_name = output_name + "-vslice_" + Patch::to_string(j) + ".txt";
			FILE *slice_file = fopen(slice_file_name.data(), "w");
			 
			for(int i = 0; i < W; i++)
			{
				for(int ch = 0; ch < f_valid_channels; ch ++)
				{
					fprintf(slice_file, "%.8f\t", host_input_f[ch * W * H + i * W + j]);
				}
				fprintf(slice_file, "\n");
			}
			fclose(slice_file);
			dprintf(0, "Input V slice %d saved in %s.\n", j, (slice_file_name ).data());						
		}
	}
	printf("\n");

	
	/* GPU info */
	
	if(debug) dprintf(0, "GPU INFORMATION\n");
	if(debug) dprintf(0, "Getting available GPU memory ...\n");
	size_t free;
	size_t total;
	cudaMemGetInfo(&free, &total);
	
	if (debug) printf("There are %lu bytes available of %lu\n", free, total);
	long unsigned int needgpu = 4 * f_valid_channels * H * W* sizeof(type) +  H * W* sizeof(type);
	if(debug) printf("allocating images memory in device, need at least %lu free bytes on GPU ... \n", needgpu);
	if(free < needgpu)
		if(debug) printf("Not enough available memory on GPU. There can be errors.\n");

	
	// ******************************************************************************************************************************
	//  ************* FILTERING 
	// ******************************************************************************************************************************

	// Pointers to real input for algorithm. Can be modified version of host_input_f
	type *host_f = host_input_f;

	valid_channels = f_valid_channels;
	nchannels = f_nchannels;

	int bit_depth = f_bit_depth;
	int rgb_format = f_rgb_format;

	// Pointer for output 
	//type *host_output = NULL;

	output_name = output_name + "-" + get_conf_name(conf);

	if(conf == CONF_GAMMIFY)
	{
		std::string final_output_name = output_name + get_im_format(output_format);
		//unsigned char* p = pixels;
		unsigned char * pixels = new unsigned char[H * W * f_valid_channels];
		
		unsigned char *p = pixels;
		for(int i = 0; i< H*W; i++)
		{
			//Change first color channels
			for(int ch = 0; ch < f_valid_channels; ch ++){
				type val = (type)(host_f[ch * H * W + i]);				
				// Apply gamma
				val = Matrix::apply_gamma(val);
				*p = (unsigned char)(val * RGB_REF);
				p++;
			}
			//Skip possible alpha channel (last channel)
			for(int ch = f_valid_channels; ch < f_nchannels; ch ++){
				p++;
			}
		}
		if(debug) dprintf(0, "done.\n");

		// Save PNG image
		dprintf(0, "\nOUTPUT PNG with GAMMA correction ...");
		write_png(final_output_name, pixels, H, W, f_nchannels, 8, debug);
		dprintf(0, "\n > %s saved.", final_output_name.data());
		if(save_log)
		{
			fprintf(log_file, "\n > %s saved.", final_output_name.data());
		}

		return 1;

	}

	// If we just want to apply gamma or take out gamma. Gamma is already taken out.
	if(conf == CONF_UNGAMMIFY)
	{
		std::string final_output_name = output_name + get_im_format(output_format);
		//unsigned char* p = pixels;
		unsigned char * pixels = new unsigned char[H * W * f_nchannels];
		
		unsigned char *p = pixels;
		for(int i = 0; i< H*W; i++)
		{
			//Change first color channels
			for(int ch = 0; ch < f_valid_channels; ch ++){
				type val = (host_f[ch * H * W + i]);
				*p = (unsigned char)(val * RGB_REF);
				p++;
			}
			//Skip possible alpha channel (last channel)
			for(int ch = f_valid_channels; ch < f_nchannels; ch ++){
				p++;
			}
		}
		if(debug) dprintf(0, "done.\n");

		// Save PNG image
		dprintf(0, "\nOUTPUT PNG with GAMMA correction ...");
		write_png(final_output_name, pixels, H, W, f_nchannels, f_bit_depth, debug);
		dprintf(0, "\n > %s saved.", final_output_name.data());
		if(save_log)
		{
			fprintf(log_file, "\n > %s saved.", final_output_name.data());
		}

		return 1;

	}

	if (conf == CONF_TO_GRAY)
	{
		type *host_f_gray = new type[H * W];
		/*Matrix::RGB_to_Gray(host_input_f, host_f_gray, H * W, f_valid_channels, f_rgb_format);*/
		for(int i = 0; i < H*W; i++)
		{
			host_f_gray[i] = 0.2989*host_input_f[3 * i ] + 0.587* host_input_f[3 * i + 1] + 0.114*host_input_f[3*i + 2];
		}
		// save to png
		
		unsigned char * pixels = new unsigned char[H * W ];

		if(print_linear)
		{
			for(int i = 0; i < H * W ; i++)
			{
				pixels[i] = (unsigned char) (host_f_gray[i] * 255);
			}
			std::string output_file_name = output_name +  "-LINEAR-" + get_im_format(output_format) ;
			dprintf(0, "\nOUTPUT PNG in LINEAR space ...");
			write_png(output_file_name , pixels, H, W, 1 , 8, debug);
			dprintf(0, "\n\t > %s saved.\n", output_file_name.data());
		}

		if(print_gamma)
		{
			std::string output_file_name = output_name + get_im_format(output_format);
			for(int i = 0; i < H * W ; i++)
			{
				pixels[i] = (unsigned char) (Matrix::apply_gamma<type>(host_f_gray[i]) * 255);
			}
			dprintf(0, "\nOUTPUT PNG with GAMMA correction ...");
			write_png(output_file_name , pixels, H, W, 1 , 8, debug);
			dprintf(0, "\n\t > %s saved.\n", output_file_name.data());
		}
		
		delete[] host_f_gray, host_input_f, pixels;
		return 1;

	}

	if(conf == CONF_RGF || CONF_ARRGF || CONF_CONV)
	{
		dprintf(0, "%s", (std::string("Configuration: ") + get_conf_name(conf) + "\n").data()) ;

		//type *host_f_gray_log = NULL;

		// If we want to make hdr tone mapping, we need to convert to gray scale as input
		if(make_hdr_tone_mapping)
		{
			dprintf(0, "\nConverting to GRAY for HDR tone mapping ... ");
			type *host_f_gray = new type[H * W];
			//host_f_gray_log = new type[H * W];

			valid_channels = 1;
			nchannels = 1;

			Matrix::RGB_to_Gray(host_input_f, host_f_gray, H * W, f_valid_channels, f_rgb_format);
			/*for(int i = 0; i < H*W; i++)
			{
				host_f_gray[i] = 0.2126*host_input_f[3 * i ] + 0.7152* host_input_f[3 * i + 1] + 0.0722*host_input_f[3*i + 2];
			}*/
			dprintf(0, "\n\tRGB to Gray calculated.");

			host_f = host_f_gray;
			
		}

		/* Allocate other arrays memory in host */
		type *host_g = new type[valid_channels * H * W];
		type *host_temp = new type[valid_channels * H * W];

		// Intialize g in host
		if(debug) dprintf(0, "\nInitializing g in Host ...");
		if(!load_it)
		{
			for(int ch = 0; ch< valid_channels ; ch++){
				//fill channel i for f
				for(int i = 0; i< H*W; i++){	
					// Initialize images in Host	
					if(conf == 3 )
					{
						host_g[ch * W * H + i] = host_f[ch * W * H + i];	// bilateral filter starts with f
					}
					else
					{
						host_g[ch * W * H + i] = (type) 0;
					}
				}
			}
		} else {
			for(int ch = 0; ch< valid_channels ; ch++){
				unsigned char *p = (gpixels +ch);
				//fill channel i for f
				for(int i = 0; i< H*W; i++)
				{	
					// Initialize images in Host		
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

		/* Allocate GPU arrays */
		type *dev_f;
		cudaMalloc((void**)&dev_f, valid_channels * H * W * sizeof(type));
		gpuErrChk( cudaPeekAtLastError() );
		if(debug) dprintf(0,"\nCopying data into device ... ");
		for(int ch = 0; ch < valid_channels; ch++ ){
			cudaMemcpy(dev_f + ch*H*W , host_f + ch * H * W, H*W*sizeof(type), cudaMemcpyHostToDevice);
			gpuErrChk( cudaPeekAtLastError() );		
		}

		/* GPU arrays */
		if(debug) printf("\nAllocating images memory in device ... ");
		type *dev_g, *dev_g_last;
		type *dev_stdev, *dev_temp;
		type *dev_temp2;
		
		type *host_local_measure = NULL;
		if(txt_local_measure || print_local_measure_gamma || print_local_measure_linear)
		{
			host_local_measure = new type[H * W];
		}
		
		dim3 ggridsize = dim3(ceil((type)W/gbsX), ceil((type)H/gbsY), 1);
		dim3 gblocksize = dim3(gbsX, gbsY,1);

		// Initialize other helpful GPU arrays
		if(adaptive)
		{
			cudaMalloc((void**)&dev_stdev, H * W * sizeof(type));
			gpuErrChk( cudaPeekAtLastError() );
		}
		cudaMalloc((void**)&dev_temp, valid_channels * H * W * sizeof(type));
		gpuErrChk( cudaPeekAtLastError() );
		
		cudaMalloc((void**)&dev_temp2, valid_channels * H * W * sizeof(type));
		gpuErrChk( cudaPeekAtLastError() );
		
	
		cudaMalloc((void**)&dev_g, valid_channels * H * W * sizeof(type));
		gpuErrChk( cudaPeekAtLastError() );

		if(calc_norms){
			cudaMalloc((void**)&dev_g_last, valid_channels * H * W * sizeof(type));
			gpuErrChk( cudaPeekAtLastError() );
		}

		// Generic output name
		bool print_it = false; // True if iteration is to be printed

		// FILTERING LOOPS:

		// Loop for scale values (scale)
		if(debug) dprintf(0, "\nStarting SCALEloop \n");
		for(int i = 0; i < (int)scaleValues.size(); i++)
		{
		
			type scale = scaleValues.at(i);
			if(debug) dprintf(0, "scale = %.2f\n", scale);

			// Scale range
			f_range *= scale;
			if(f_range == 0) f_range = 1.0;
			

			if(debug) dprintf(0, "Scaling input for scale = %.2f", scale);
			for(int i = 0; i<valid_channels; i++)
			{
				scale_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_temp + i * H * W,  scale, H, W);
				gpuErrChk( cudaPeekAtLastError() );
			}
			std::swap(dev_f, dev_temp);
			

		// Loop for spatial sigma values (ss)
		if(debug) dprintf(0, "\nStarting SS loop");
		for(int i = 0; i < (int)ssValues.size(); i++)
		{		
			ss = ssValues.at(i);
			if(debug) dprintf(0, "ss = %.2f\n", ss);
		
			if(!sl_exp) sl = ss;
				
			// Calculate size of regions
			switch(sker)
			{
				case 0:
					sskh = ceil(gaussian_neigh * ss);
					sskw = ceil(gaussian_neigh * ss);
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
			sreg = sregValues.at(i);
			if(debug) dprintf(0, "sreg = %.2f\n", sreg);

			switch(regker)
			{
				case 0:
					regkh = ceil(gaussian_neigh * sreg * ss);
					regkw = ceil(gaussian_neigh * sreg * ss);
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
					lmreg_kh = ceil(gaussian_neigh * lmreg_s);
					lmreg_kw = ceil(gaussian_neigh * lmreg_s);
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
					lkh = ceil(gaussian_neigh * sl);
					lkw = ceil(gaussian_neigh * sl);
					break;
				// circle characteristic weights
				case 0:
					lkh = ceil(gaussian_neigh * sl);
					lkw = ceil(gaussian_neigh * sl);
					break;
				default:
					lkh = ceil(gaussian_neigh * sl);
					lkw = ceil(gaussian_neigh * sl);
					break;
			}
			if(debug) dprintf(0, "lweights region for gpu: lkh = %i, lkw = %i\n", lkh, lkw);

		if(debug) dprintf(0, "\nStarting SR loop ...");		
		for(int i = 0; i < (int)srValues.size(); i++)
		{		
			sr = srValues.at(i);
			
			// ************** PROCESS for given parameters ***************
			
			// Start convergence values over
			for(int i = 0; i < (int)show_norm_list.size(); i++)
			{
				show_conv_values.at(i) = conv_eps_list.size();
				conv_values.at(i) = conv_eps_list.size();	
			}

			for(int i = 0; i < (int)show_norm_list.size(); i++)
			{
				if(isInVector<std::string>(conv_norm_list, show_norm_list.at(i)) < 0)
				{
					conv_values.at(i) = 0;
				}
			}
			
		dprintf(0, "\n\n\tWorking for ss = %.2f, sr = %.2f, sreg = %.2f, scale = %.2f\n", ss, sr, sreg, scale);
		if(save_log) fprintf(log_file, "\n\n\tWorking for ss = %.2f, sr = %.2f, scale = %.2f\n", ss, sr, scale);
		
		//Initialize dev_g according to conf
		if(debug) dprintf(0, "\nIntializing dev_g in device (copying from g_host)... ");
		cudaMemcpy(dev_g, host_g, valid_channels * H * W * sizeof(type), cudaMemcpyHostToDevice);				
		if(debug) dprintf(0, "done.");	
	
		// Perform iterations
		bool run = true;
		
		it = it_from;
			
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
				cudaMemcpy(dev_g_last, dev_g, H * W * valid_channels * sizeof(type), cudaMemcpyDeviceToDevice);
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
					
						for(int i = 0; i < valid_channels; i++)
						{
							// stdv is calculated in 3d using euclidean metric in rgb space
							//subs_kernel_rgb<type><<<ggridsize, gblocksize>>>(dev_f, dev_g, devtemp, valid_channels, H, W); -> dev_temp_i
							subs_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_g + i * H * W, dev_temp + i * H * W, H, W);
							gpuErrChk( cudaPeekAtLastError() );
							
							// Calculate mean value for each channel. -> dev_temp2
							//local_mean_kernel<type><<<ggridsize, gblocksize>>>(dev_temp + i * H * W, dev_temp2 + i * H * W, H, W, sl, lweights, lkh, lkw);
							//gpuErrChk( cudaPeekAtLastError() );
				
						}
						//centered_max_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_temp2, dev_stdev, H, W, valid_channels, sl, lweights, lkh, lkw);
						max_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_stdev, H, W, valid_channels, sl, lweights, lkh, lkw);
						gpuErrChk( cudaPeekAtLastError() );						
						cudaDeviceSynchronize();
						
						break;
			
					case 1:	// Mean as local measure
					
						for(int i = 0; i < valid_channels; i++)
						{
							// stdv is calculated in 3d using euclidean metric in rgb space
							//subs_kernel_rgb<type><<<ggridsize, gblocksize>>>(dev_f, dev_g, devtemp, valid_channels, H, W); -> dev_temp_i
							subs_kernel<type><<<ggridsize, gblocksize>>>(dev_f + i * H * W, dev_g + i * H * W, dev_temp + i * H * W, H, W);
							gpuErrChk( cudaPeekAtLastError() );
							
							// Calculate mean value for each channel. -> dev_temp2
							//local_mean_kernel<type><<<ggridsize, gblocksize>>>(dev_temp + i * H * W, dev_temp2 + i * H * W, H, W, sl, lweights, lkh, lkw);
							//gpuErrChk( cudaPeekAtLastError() );
				
						}
						mean_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_stdev, H, W, valid_channels, sl, lweights, lkh, lkw);
						//centered_mean_dist_kernel<type><<<ggridsize, gblocksize>>>(dev_temp, dev_temp2, dev_stdev, H, W, valid_channels, sl, lweights, lkh, lkw);
						gpuErrChk( cudaPeekAtLastError() );						
						cudaDeviceSynchronize();
						
						break;
					

					
					default: // 2 or default: Standard deviation as local measure
					
						for(int i = 0; i < valid_channels; i++)
						{
							// stdv is calculated in 3d using euclidean metric in rgb space
							//subs_kernel_rgb<type><<<ggridsize, gblocksize>>>(dev_f, dev_g, devtemp, valid_channels, H, W); -> dev_temp_i
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
						sum_channels_kernel<type><<<ggridsize, gblocksize>>>(dev_temp2, dev_stdev, H, W, valid_channels);
						gpuErrChk( cudaPeekAtLastError() );
						
						// Normalize
						scale_in_place_kernel<type><<<ggridsize, gblocksize>>>(dev_stdev, 1.0 / ((type)valid_channels), H, W);
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

				for(int ch = 0; ch<valid_channels; ch++)
				{	
					trilateral_kernel_rgb<type><<<dim3(ceil((type)W/blockConvW), ceil((type)H/blockConvH), 1), dim3(blockConvW, blockConvH,1)>>>(dev_g + ch * H * W, NULL, dev_temp + ch * H * W, 0, H, W, domain_extension, sreg * ss, 0, 0, regker, -1, -1, regkh, regkw);		
					//convolution_kernel<type><<<dim3(ceil((type)W/blockConvW), ceil((type)H/blockConvH), 1), dim3(blockConvW, blockConvH,1)>>>(dev_g + i * H * W, dev_temp + i * H * W,  H, W, sreg * ss , regker, regkh, regkw);
					gpuErrChk( cudaPeekAtLastError() );
				}
				std::swap(dev_g, dev_temp);
				// dev_temp is free now
			}
			cudaDeviceSynchronize();
			
			// Joint Bilateral Filter
			dprintf(0, "b\t");
			for(int ch = 0; ch<valid_channels; ch++)
			{	
				type* devf = dev_f;
				if(conf == CONF_BIL)
				{
					// Bilateral algorithm do not work on original f
					devf = dev_g;
				}
				if (adaptive)
				{
					trilateral_kernel_rgb<type><<<dim3(ceil((type)(W)/blockBilW), ceil((type)(H)/blockBilH),1), dim3(blockBilW, blockBilH,1)>>>(devf + ch * H * W, dev_g, dev_temp + ch * H * W, valid_channels, H, W, domain_extension, ss, sr / sr_ref, dev_stdev, infsr, sm, sker, rker, mker, sskh, sskw);
				}
				else
				{
					trilateral_kernel_rgb<type><<<dim3((int)(W/blockBilW) + 1, (int)(H/blockBilH) + 1,1), dim3(blockBilW, blockBilH,1)>>>(devf + ch * H * W, dev_g, dev_temp + ch * H * W, valid_channels, H, W, domain_extension, ss, sr * f_range, sm, sker, rker, mker, sskh, sskw);
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
						dist2_rgb_kernel<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, valid_channels);
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
						l2loc2_kernel_rgb<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, valid_channels, sskh, sskw);
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
						l2loc2_kernel_rgb<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, valid_channels, 0, 0);
						//dist2_rgb_kernel<type><<<dim3(ceil((type)W/blockBilW), ceil((type)H/blockBilH), 1), dim3(blockBilW, blockBilH,1)>>>(dev_g, dev_g_last, dev_temp, H, W, valid_channels);
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
						if( norm_val <= eps * eps * valid_channels)
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
			else if(it > 0 && isInVector<int>(it_list, it, 0) >= 0)
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
				
				// Copy back result of iteration to Host en host_temp
				if (debug) dprintf(0, "\nGetting back result to host , valid_channels = %d... ", valid_channels);
				for(int ch = 0; ch < valid_channels; ch++)
				{		
					cudaMemcpy( host_temp + ch * H * W, dev_g + ch * H * W, H * W * sizeof(type), cudaMemcpyDeviceToHost );
					gpuErrChk( cudaPeekAtLastError() );
					
				}

				// Make output generic name
				std::string out_name = output_name;

				make_output_name(&out_name);

				// Print iteration result with GAMMA correction
				if(print_gamma)
				{
					dprintf(0, "\nPrinting iteration %d with GAMMA	correction ...", it);
					std::string final_output_file_name = out_name + get_im_format(output_format);
					//unsigned char* p = pixels;
					unsigned char * pixels = new unsigned char[H * W * nchannels];
					
					unsigned char *p = pixels;
					for(int i = 0; i< H*W; i++)
					{
						//Change first color channels
						for(int ch = 0; ch < valid_channels; ch ++){
							type val = ((type)(host_temp[ch * H * W + i]));
							
							if(scale_back)
								val = val / scale;
							
							val = Matrix::apply_gamma<type>(val);

							*p = (unsigned char)(val * 255);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = valid_channels; ch < nchannels; ch ++){
							*p = 0;
							p++;
						}
					}
					if(debug) dprintf(0, "done.");

					// Save PNG image
					write_png(final_output_file_name, pixels, H, W, nchannels, 8, debug);
					dprintf(0, "\n\t > %s", final_output_file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\n\t > %s", final_output_file_name.data());
					}
					delete[] pixels;
				}

				// If we want to print linear space and default ouput was not in linear space
				if(print_linear)
				{
					dprintf(0, "\nPrinting iteration %d in LINEAR space ...", it);
					std::string final_output_file_name = out_name + "-linear.png";
					unsigned char *pixels = new unsigned char[H * W * nchannels];
					unsigned char* p = pixels;
					for(int i = 0; i< H*W; i++)
					{
						//Change first color channels
						for(int ch = 0; ch < valid_channels; ch ++){
							type val = ((type)(host_temp[ch * H * W + i]));
							
							if(scale_back)
								val = val / scale;
							*p = (unsigned char)(val * RGB_REF);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = valid_channels; ch < nchannels; ch ++){
							*p = 0;
							p++;
						}
					}

					// Save PNG image
					dprintf(0, "\nOUTPUT PNG in LINEAR space ...");
					write_png(final_output_file_name, pixels, H, W, nchannels, 8, debug);
					dprintf(0, " \n\t > %s", final_output_file_name.data());
					
					if(save_log)
					{
						fprintf(log_file, "\n\t > %s", final_output_file_name.data());
					}
					delete[] pixels;
				}

				// HDR TONE MAPPING: make back RGB and print it
				if(make_hdr_tone_mapping)
				{
					dprintf(0, "\nMaking Tone Mapping ...");
					std::string output_tm_name = out_name + "-TONE_MAPPED.png";
					
					if(input_format == IMAGE_HDR)
					{
						int output_nchannels = 3;
						type * host_output = new type[H * W * output_nchannels ];
						int output_rgb_format = MATRIX_RGB_TYPE_CHUNK;
						switch(f_rgb_format)
						{
							case MATRIX_RGB_TYPE_INTER:
								for(int i = 0; i < H * W; i++)
								{
									for(int c = 0; c < output_nchannels; c++)
									{		
										host_output[i + c * H * W] = (type)((double)host_input_f[3 * i + c] * exp( (double)0.25f * log((double)host_temp[i] + (double)0.00001)  + log((double)host_f[i] + 0.0001) - log( (double)host_temp[i] + (double)0.0001) ) / (double)host_f[i] );
									}							
								}
								break;
							case MATRIX_RGB_TYPE_CHUNK:
								for(int i = 0; i < H * W; i++)
								{
									for(int c = 0; c < output_nchannels; c++)
									{	
										host_output[i + c * H * W] = (type)((double)host_input_f[i + c * H * W] * exp( (double)0.25f * log((double)host_temp[i] + (double)0.00001)  + log((double)host_f[i] + 0.0001) - log( (double)host_temp[i] + (double)0.0001) ) / (double)host_f[i] );
									}															
								}
							break;
						}

						// Calculate min value to correct bias
						type min_val = host_output[0];
						for(int i = 0; i < H * W * output_nchannels ; i++)
						{
							min_val = min(min_val, host_output[i]);
						}
						
						// Correc bias
						for(int i = 0; i < H * W * output_nchannels; i++)
						{
							host_output[i] -= min_val;
							host_output[i] *= 0.45;
						}
						
						dprintf(0, "\nPrinting TONE MAPPED PNG with GAMMA correction ...");

						/*unsigned char *pixels = new unsigned char[H * W * output_nchannels];
						unsigned char* p = pixels;
						for(int i = 0; i< H*W; i++)
						{
							//Change first color channels
							for(int ch = 0; ch < output_nchannels; ch ++){
								type val = ((type)(host_temp[ch * H * W + i]));
								
								if(scale_back)
									val = val / scale;
								*p = (unsigned char)(val * RGB_REF);
								p++;
							}
						}*/

						// Apply gamma
						Matrix::apply_gamma<type>(host_output, host_output, H * W * output_nchannels);

						write_png_float<type>(output_tm_name, host_output, H, W, output_nchannels, 8, debug);
						dprintf(0, "\n\t > %s", output_tm_name.data());

						if(save_log)
						{
							fprintf(log_file, "\n\t > %s\n", output_tm_name.data());
						}

						//delete[] pixels;
						delete[] host_output;
						//delete[] host_detail;

					}

				}
				// Print TXT local measure
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

				// Print PNG LOCAL MEASURE with gamma correction
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
					
					write_png(file_name, pixels, H, W, 1, 8, debug);
					dprintf(0, "\n\t> %s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\nLocal Measure with Gamma > %s saved.", file_name.data());
					}
					delete[] pixels;
				
				}
				// Print LOCAL MEASURE in LINEAR space
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

					
					write_png(file_name, pixels, H, W, 1, 8, debug);
					dprintf(0, "\n\t> %s saved.", file_name.data());
					if(save_log)
					{
						fprintf(log_file, "\nLocal Measure Linear > %s saved.", file_name.data());
					}
					delete[] pixels;
				
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
						for(int ch = 0; ch < valid_channels; ch ++){
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
						for(int ch = valid_channels; ch < nchannels; ch ++){
							p++;
						}
					}
					if(debug) dprintf(0, "done.\n");

					// Save PNG image
					dprintf(0, "\nPrinting DIFF PNG with GAMMA correction ... \n");
					write_png(file_name, pixels, H, W, nchannels, bit_depth, debug);
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
						for(int ch = 0; ch < valid_channels; ch ++){
							type val = abs(host_temp[ch * H * W + i] - host_f[ch * H * W + i]);
							
							if(scale_back)
								val = val / scale;
							*p = (unsigned char)(val * RGB_REF);
							p++;
						}
						//Skip possible alpha channel (last channel)
						for(int ch = valid_channels; ch < nchannels; ch ++){
							p++;
						}
					}

					// Save PNG image
					dprintf(0, "Printing DIFF PNG in LINEAR SPACE ...\n");
					write_png(file_name, pixels, H, W, nchannels, bit_depth, debug);
					dprintf(0, "\n\t > %s", file_name.data());
					
					if(save_log)
					{
						fprintf(log_file, "\n\t > %s", file_name.data());
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
						for(int ch = 0; ch < valid_channels; ch ++){
							type val = abs((host_temp[ch * H * W + i] - host_f[ch * H * W + i]));
							sum += val * val;	
						}
						sum = sqrt(sum / valid_channels);
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
					write_png(file_name, pixels, H, W, 1, 8, debug);
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
						for(int ch = 0; ch < valid_channels; ch ++){
							type val = abs((host_temp[ch * H * W + i] - host_f[ch * H * W + i]));	
							sum += val * val;	
						}
						sum = sqrt(sum / valid_channels);
						if(scale_back)
								sum = sum / scale;
						*p = (unsigned char)(sum * RGB_REF);
						p++;
					}
					if(debug) dprintf(0, "done.\n");

					// Save PNG image
					dprintf(0, "\nPrinting DIFF SINGLE image in LINEAR space ...");
					write_png(file_name, pixels, H, W, 1, 8, debug);
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
					dprintf(0, "\nSaving TXT version in LINEAR space. It could generate a heavy file. Alpha channel will NOT be saved.");
					std::string final_output_name = out_name;
					Util::printToTxt<type>(final_output_name, host_temp, H, W, valid_channels);
					if(save_log)
					{
						fprintf(log_file, "\n\t > %s", (final_output_name).data());
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
							for(int ch = 0; ch < valid_channels; ch ++)
							{
								fprintf(slice_file, "%.8f\t", host_temp[ch * W * H + i * W + j]);
							}
							fprintf(slice_file, "\n");
						}
						fclose(slice_file);
						dprintf(0, "\n\tH SLICE %d saved in %s.", i, (slice_file_name ).data());						
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
							for(int ch = 0; ch < valid_channels; ch ++)
							{
								fprintf(slice_file, "%.8f\t", host_temp[ch * W * H + i * W + j]);
							}
							fprintf(slice_file, "\n");
						}
						fclose(slice_file);
						dprintf(0, "\n\tV SLICE %d saved in %s.", j, (slice_file_name ).data());						
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
							for(int ch = 0; ch < valid_channels; ch ++){
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
							for(int ch = valid_channels; ch < nchannels; ch ++){
								p++;
							}
						}
						if(debug) dprintf(0, "done.\n");

						// Save PNG image
						dprintf(0, "\n Printing CONTRAST ENHANCEMENT PNG with GAMMA correction ...");
						write_png(file_name, pixels, H, W, nchannels, bit_depth, debug);
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
							for(int ch = 0; ch < valid_channels; ch ++){
								int ind = ch * H * W + i;
								type val = host_temp[ind] + ce_factor * ( host_f[ind] - host_temp[ind]) ;
								
								if(scale_back)
									val = val / scale;
								*p = (unsigned char)(val * RGB_REF);
								p++;
							}
							//Skip possible alpha channel (last channel)
							for(int ch = valid_channels; ch < nchannels; ch ++){
								p++;
							}
						}
						if(debug) dprintf(0, "done.\n");

						// Save PNG image
						dprintf(0, "\n Printing CONTRAST ENHANCEMENT PNG in LINEAR space ...");
						write_png(file_name, pixels, H, W, nchannels, bit_depth, debug);
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
	}
	
	if(save_log)
		fclose(log_file);
	
	printf("\n");
	return 1;
	
}


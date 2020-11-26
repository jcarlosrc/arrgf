#include <stdio.h>
#include <string>

bool read_png(const char* filename, int &height, int &width, int& nchannels, unsigned char* &pixels, unsigned char &color_type, unsigned char &bit_depth, int& nch_no_alpha, bool debug = false);
//bool read_png(std::string input, int &height, int &width, int& nchannels, unsigned char* &pixels, unsigned char &color_type, unsigned char &bit_depth, int& nch_no_alpha);
bool save_image(const char* file_name, unsigned char *pixels, int height, int width, int nchannels, unsigned char color_type, unsigned char bit_depth, bool debug = false);
//bool save_image(std::string output, unsigned char *pixels, int height, int width, int nchannels, unsigned char color_type, unsigned char bit_depth);


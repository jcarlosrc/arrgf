#ifndef __PNG_TOOL_H__
#define __PNG_TOOL_H__

#include <png.h>
#include <stdio.h>
#include <string>

unsigned char get_png_color_type(int nchannels)
{
    switch (nchannels)
    {
    case 1:
        return PNG_COLOR_TYPE_GRAY;
        break;
    case 2:
        return PNG_COLOR_TYPE_GA;
        break;
    case 3:
        return PNG_COLOR_TYPE_RGB;
        break;
    case 4:
        return PNG_COLOR_TYPE_RGB_ALPHA;
        break;
    default:
        return PNG_COLOR_TYPE_RGB;
        break;
    }
}

int get_nchannels(unsigned char png_color_type)
{
    switch (png_color_type)
    {
    case PNG_COLOR_TYPE_GRAY:
        return 1;
        break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        return 2;
        break;
    case PNG_COLOR_TYPE_RGB:
        return 3;
        break;
    case PNG_COLOR_TYPE_RGBA:
        return 4;
        break;
    default:
        return 3;
        break;
    }
}

bool write_png(const char* file_name, unsigned char *pixels, int height, int width, int nchannels, unsigned char bit_depth, bool debug)
{
    unsigned char color_type = get_png_color_type(nchannels);
    if(debug)
    {
        dprintf(0, "\nwrite_png: file_name = %s", file_name);
        dprintf(0, "\nwrite_png: pixels = %i", *pixels);
        dprintf(0, "\nwrite_png: height = %d, width = %d", height, width);
        dprintf(0, "\nwrite_png: nchannels = %i", nchannels);
        dprintf(0, "\nwrite_png: color_type = %u", color_type);
        dprintf(0, "\nwrite_png: bit_depth = %u", bit_depth);
    }
    /* create file */
    FILE *fp = fopen(file_name, "wb");
    if (!fp){
            printf("[write_png_file] File %s could not be opened for writing", file_name);
            return false;
            }

    /* initialize stuff */
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr){
            printf("[write_png_file] png_create_write_struct failed");
            return false;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
            printf("[write_png_file] png_create_info_struct failed");
            return false;
            }
    /* handling errors */
    if (setjmp(png_jmpbuf(png_ptr))){
            printf("[write_png_file] Error during init_io");
            return false;
            }

    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr))){
            printf("[write_png_file] Error during writing header");
            return false;
            }

    png_set_IHDR(png_ptr, info_ptr, width, height,
                    bit_depth, color_type, PNG_INTERLACE_NONE,
                    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr))){
            printf("[write_png_file] Error during writing bytes");
            return false;
            }
    
    //Create row pointers to data in 'pixels'
    unsigned char** row_pointers = (unsigned char**) malloc(sizeof(unsigned char*) * height);
    for (int i=0; i<height; i++)
            row_pointers[i] = &pixels[width * i*nchannels];

    png_write_image(png_ptr, row_pointers);

    /* end write */
    if (setjmp(png_jmpbuf(png_ptr))){
        printf("[write_png_file] Error during end of write");
        return false;
    }
    png_write_end(png_ptr, NULL);

    free(row_pointers);

    fclose(fp);

    dprintf(0, "\n\t > %s", file_name);

    return true;
}

template<typename type> bool write_png_float(const std::string file_name, type *pixels, int height, int width, int nchannels, unsigned char bit_depth = 8, bool debug = false)
{
    unsigned char *pixels_uchar = new unsigned char[width * height * nchannels];
    for(int i = 0; i < height * width; i ++)
    {
        for(int c = 0; c < nchannels; c++)
        {
            pixels_uchar[nchannels *i + c] = (unsigned char)(pixels[i + c * width * height] * 255);
        }
        
    }
    bool a = write_png(file_name.data(), pixels_uchar, height, width, nchannels, bit_depth, debug);
    delete[] pixels_uchar;
    return a;
}

bool write_png(const std::string file_name, unsigned char *pixels, int height, int width, int nchannels, unsigned char bit_depth, bool debug )
{
    if(debug)
    {
        dprintf(0, "\nwrite_png (string version): file_name = %s", file_name.data());
        dprintf(0, "\nwrite_png (string version): pixels = %i", *pixels);
        dprintf(0, "\nwrite_png (string version): height = %d, width = %d", height, width);
        dprintf(0, "\nwrite_png (string version): nchannels = %i", nchannels);
        unsigned char color_type = get_png_color_type(nchannels);
        dprintf(0, "\nwrite_png (string version): color_type = %u", color_type);
        dprintf(0, "\nwrite_png (string version): bit_depth = %u", bit_depth);
    }

    return write_png(file_name.data(), pixels, height, width, nchannels, bit_depth, debug);

}
unsigned char get_png_color_type(int nchannels);
int get_nchannels(unsigned char png_color_type);

bool read_png(const char* filename, unsigned char* &pixels, int &height, int &width, int& nchannels,  unsigned char &bit_depth, bool debug)
{
    unsigned char color_type;
    FILE *file = fopen(filename, "rb");
    if(file == NULL)
        return 0;
    //check is png signature is present
    unsigned char sig[8];
    if(fread(sig, 1, sizeof(sig), file) < 8){
        fclose(file);
        return false;
    }
    if(png_sig_cmp(sig, 0, 8)){
        fclose(file);
        return false;
    }

    if(debug) printf("\ncreate to data structures: png_struct and png_infop");
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(png == NULL){
    	printf("\nCould not create png structure");
        fclose(file);
        return false;
    }
    png_infop info = png_create_info_struct(png);
    if(info == NULL){
    	printf("\nCould not create info structure");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(file);
        return false;
    }

    //printf("Set libpng error handling mechanism\n");
    if (setjmp(png_jmpbuf(png))){
    	printf("\nError");
        png_destroy_read_struct(&png, &info, NULL);
        fclose(file);
        //require input pixels pointer to be null
        if(pixels != NULL){
            delete[] pixels;
            pixels = NULL;
        }
        return false;
    }

    //printf("pass open file to png structures\n");
    png_init_io(png, file);

    //printf("skip signature reads, we have already done that\n");
    png_set_sig_bytes(png, sizeof(sig));

    //printf("get image information");
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    if(debug) printf("\nheight = %i, width = %i", height, width);

    //printf("set least one byte per channel\n");
    if(png_get_bit_depth(png, info)  < 8){
        png_set_packing(png);
    }

    //printf("if transparency, convert it to alpha\n");
    if(png_get_valid(png, info, PNG_INFO_tRNS)){
        png_set_tRNS_to_alpha(png);
    }

    if(debug) printf("\ncolor type: ");
    color_type = png_get_color_type(png, info);
    switch(color_type){
    	case PNG_COLOR_TYPE_GRAY:
    		nchannels = 1;
    		if(debug) printf("%u, PNG_COLOR_TYPE_GRAY", color_type);
    		break;
    	case PNG_COLOR_TYPE_GRAY_ALPHA:
    		nchannels = 2;
    		if(debug) printf("%u, PNG_COLOR_TYPE_GRAY_ALPHA", color_type);
    		break;
    	case PNG_COLOR_TYPE_RGB:
    		nchannels = 3;
    		if(debug) printf("%u, PNG_COLOR_TYPE_RGB", color_type);
    		break;
    	case PNG_COLOR_TYPE_RGBA:
    		nchannels = 4;
    		if(debug) printf("%u, PNG_COLOR_TYPE_RGBA", color_type);
    		break;
  		default:
  			printf("\npng_tool: Not supported color type.");
  			return false;
    }
    
    bit_depth = png_get_bit_depth(png, info);
    if(debug) printf("\nbit_depth = %i", bit_depth);
    
    unsigned char bytespp = (unsigned char)(png_get_rowbytes(png, info) / width);
	if(debug) printf("\nbytespp = %i", bytespp);
	
    png_set_interlace_handling(png);

    png_read_update_info(png, info);

    if(debug) printf("\nallocate pixel buffer to save pixel values\n");
    pixels = new unsigned char [height * width * nchannels];
    //printf("setup array with row pointers into pixel buffer\n");
    png_bytep rows[height];
    unsigned char *p = pixels;
    for(int i = 0; i < height; i++){
        rows[i] = p;
        p += width * nchannels;
    }

    if(debug) printf("\nread all rows (data goes into 'pixels' buffer)");
    //note that all encoding error will jump into the setjmp pointers
    //and eventually become false
    png_read_image(png, rows);

    if(debug) printf("\nread the end of the png file");
    png_read_end(png, NULL);
    if(debug) printf("\nfinally, clean up and return true");
    png_destroy_read_struct(&png, &info, NULL);
    if(debug) printf("\nclose file");
    fclose(file);
    if(debug) printf("\nReading image done.");
    return true;
}

bool read_png(const std::string input, unsigned char* &pixels, int &height, int &width, int& nchannels, unsigned char &bit_depth, bool debug)
{
	return read_png(input.data(), pixels, height, width, nchannels, bit_depth, debug);
}




#endif


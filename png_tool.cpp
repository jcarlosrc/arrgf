#include <png.h>
#include <stdio.h>
#include<algorithm>
#include <cstdio>
#include <cmath>
#include <string>

bool read_png(const char* filename, int &height, int &width, int& nchannels, unsigned char* &pixels, unsigned char &color_type, unsigned char &bit_depth, int& nch_no_alpha, bool debug = false)
{
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

    if(debug) printf("create to data structures: png_struct and png_infop\n");
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(png == NULL){
    	printf("Could not create png structure\n");
        fclose(file);
        return false;
    }
    png_infop info = png_create_info_struct(png);
    if(info == NULL){
    	printf("Could not create info structure\n");
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(file);
        return false;
    }

    //printf("Set libpng error handling mechanism\n");
    if (setjmp(png_jmpbuf(png))){
    	printf("Error\n");
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
    if(debug) printf("height = %i, width = %i\n", height, width);

    //printf("set least one byte per channel\n");
    if(png_get_bit_depth(png, info)  < 8){
        png_set_packing(png);
    }

    //printf("if transparency, convert it to alpha\n");
    if(png_get_valid(png, info, PNG_INFO_tRNS)){
        png_set_tRNS_to_alpha(png);
    }

    if(debug) printf("color type: ");
    color_type = png_get_color_type(png, info);
    switch(color_type){
    	case PNG_COLOR_TYPE_GRAY:
    		nchannels = 1;
    		nch_no_alpha = 1;
    		if(debug) printf("%u, PNG_COLOR_TYPE_GRAY\n", color_type);
    		break;
    	case PNG_COLOR_TYPE_GRAY_ALPHA:
    		nchannels = 2;
    		nch_no_alpha = 1;
    		if(debug) printf("%u, PNG_COLOR_TYPE_GRAY_ALPHA\n", color_type);
    		break;
    	case PNG_COLOR_TYPE_RGB:
    		nchannels = 3;
    		nch_no_alpha = 3;
    		if(debug) printf("%u, PNG_COLOR_TYPE_RGB\n", color_type);
    		break;
    	case PNG_COLOR_TYPE_RGBA:
    		nchannels = 4;
    		nch_no_alpha = 3;
    		if(debug) printf("%u, PNG_COLOR_TYPE_RGBA\n", color_type);
    		break;
  		default:
  			printf("png_tool: Not supported color type.\n");
  			return false;
    }
    
    bit_depth = png_get_bit_depth(png, info);
    if(debug) printf("bit_depth = %i\n", bit_depth);
    
    unsigned char bytespp = (unsigned char)(png_get_rowbytes(png, info) / width);
	if(debug) printf("bytespp = %i\n", bytespp);
	
    png_set_interlace_handling(png);

    png_read_update_info(png, info);

    if(debug) printf("allocate pixel buffer to save pixel values\n");
    pixels = new unsigned char [height * width * nchannels];
    //printf("setup array with row pointers into pixel buffer\n");
    png_bytep rows[height];
    unsigned char *p = pixels;
    for(int i = 0; i < height; i++){
        rows[i] = p;
        p += width * nchannels;
    }

    if(debug) printf("read all rows (data goes into 'pixels' buffer)\n");
    //note that all encoding error will jump into the setjmp pointers
    //and eventually become false
    png_read_image(png, rows);

    if(debug) printf("read the end of the png file\n");
    png_read_end(png, NULL);
    if(debug) printf("finally, clean up and return true\n");
    png_destroy_read_struct(&png, &info, NULL);
    if(debug) printf("close file\n");
    fclose(file);
    if(debug) printf("Reading image done.\n");
    return true;
}

/*bool read_png(std::string input, int &height, int &width, int& nchannels, unsigned char* &pixels, unsigned char &color_type, unsigned char &bit_depth, int& nch_no_alpha)
{
	return read_png(input.data(), height, width, nchannels, pixels, color_type, bit_depth, nch_no_alpha);
}*/


bool save_image(const char* file_name, unsigned char*pixels, int height, int width, int nchannels, unsigned char color_type, unsigned char bit_depth, bool debug = false)
{
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
        return true;
}
/*bool save_image(std::string output, unsigned char *pixels, int height, int width, int nchannels, unsigned char color_type, unsigned char bit_depth)
{
	return save_image(output.data(), pixels, height, width, nchannels, color_type, bit_depth);
}*/

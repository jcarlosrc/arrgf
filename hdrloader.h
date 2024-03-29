
/***********************************************************************************
	Created:	17:9:2002
	FileName: 	hdrloader.h
	Author:		Igor Kravtchenko
	
	Info:		Load HDR image and convert to a set of float32 RGB triplet.
************************************************************************************/

#include <math.h>
#include <memory.h>
#include <stdio.h>

#ifndef HDR_LOADER_H
#define HDR_LOADER_H

typedef unsigned char RGBE[4];
#define R			0
#define G			1
#define B			2
#define E			3

#define  MINELEN	8				// minimum scanline length for encoding
#define  MAXELEN	0x7fff			// maximum scanline length for encoding

static bool decrunch(RGBE *scanline, int len, FILE *file);
static bool oldDecrunch(RGBE *scanline, int len, FILE *file);

template<typename type> class HDRLoaderResult{
public:
	int width, height;
	char eMax, eMin;
	// each pixel takes 3 float32, each component can be of any value...
	type *cols;
	int *cols_2;

};

class HDRLoader {
public:
	template<typename type> static bool load(const char *fileName, HDRLoaderResult<type> &res);
};

template<typename type> type convertComponent(int expo, int val)
{
	if( expo == -128 ) return 0.0;
	type v = val / 256.0f;
	type d = (type) pow(2.0f, expo);
	return v * d;
}

template<typename type> void workOnRGBE(RGBE *scan, HDRLoaderResult<type> &res, type *cols )
{

	int len = res.width;

	while (len-- > 0) {
		int expo = scan[0][E] - 128;
		if( expo > res.eMax ) res.eMax = expo;
		if( expo != -128 && expo < res.eMin ) res.eMin = expo;
		cols[0] = convertComponent<type>(expo, scan[0][R]);
		cols[1] = convertComponent<type>(expo, scan[0][G]);
		cols[2] = convertComponent<type>(expo, scan[0][B]);
		cols += 3;
		scan++;
	}
}

template<typename type> bool HDRLoader::load(const char *fileName, HDRLoaderResult<type> &res)
{
	int i;
	char str[200];
	FILE *file;

	file = fopen(fileName, "rb");
	if (!file)
		return false;

	fread(str, 10, 1, file);
	if (memcmp(str, "#?RADIANCE", 10)) {

		fseek(file, 0, SEEK_SET);
		fread(str, 6, 1, file);
		if (memcmp(str, "#?RGBE", 6)) {
			fclose(file);
			return false;
		}
	}

	fseek(file, 1, SEEK_CUR);

	char cmd[200];
	i = 0;
	char c = 0, oldc;
	while(true) {
		oldc = c;
		c = fgetc(file);
		if (c == 0xa && oldc == 0xa)
			break;
		cmd[i++] = c;
	}
	// Get resolution
	char reso[200];
	i = 0;
	while(true) {
		c = fgetc(file);
		printf("%c", c);
		reso[i++] = c;
		if (c == 0xa)
			break;
	}
	// Apply resolution
	int w, h;
	if (!sscanf(reso, "-Y %d +X %d", &h, &w)) {
		fclose(file);
		printf("Debug: returned false");
		return false;
	}		
	res.width = w;
	res.height = h;
	
	
	printf("Debug: %d x %d", res.width, res.height);

	type *cols = new type[w * h * 3];
	res.cols = cols;
	res.eMin = 127;
	res.eMax = -127;

	RGBE *scanline = new RGBE[w];
	if (!scanline) {
		fclose(file);
		return false;
	}

	// convert image
	for (int y = h - 1; y >= 0; y--) {
		if (decrunch(scanline, w, file) == false)
			break;
		workOnRGBE<type>(scanline, res, cols );
		cols += w * 3;
	}

	delete [] scanline;
	fclose(file);

	return true;
}




bool decrunch(RGBE *scanline, int len, FILE *file)
{
	int  i, j;

	if (len < MINELEN || len > MAXELEN)
		return oldDecrunch(scanline, len, file);

	i = fgetc(file);
	if (i != 2) {
		fseek(file, -1, SEEK_CUR);
		return oldDecrunch(scanline, len, file);
	}

	scanline[0][G] = fgetc(file);
	scanline[0][B] = fgetc(file);
	i = fgetc(file);

	if (scanline[0][G] != 2 || scanline[0][B] & 128) {
		scanline[0][R] = 2;
		scanline[0][E] = i;
		return oldDecrunch(scanline + 1, len - 1, file);
	}

	// read each component
	for (i = 0; i < 4; i++) {
	    for (j = 0; j < len; ) {
			unsigned char code = fgetc(file);
			if (code > 128) { // run
			    code &= 127;
			    unsigned char val = fgetc(file);
			    while (code--)
					scanline[j++][i] = val;
			}
			else  {	// non-run
			    while(code--)
					scanline[j++][i] = fgetc(file);
			}
		}
    }

	return feof(file) ? false : true;
}

bool oldDecrunch(RGBE *scanline, int len, FILE *file)
{
	int i;
	int rshift = 0;

	while (len > 0) {
		scanline[0][R] = fgetc(file);
		scanline[0][G] = fgetc(file);
		scanline[0][B] = fgetc(file);
		scanline[0][E] = fgetc(file);
		if (feof(file))
			return false;

		if (scanline[0][R] == 1 &&
			scanline[0][G] == 1 &&
			scanline[0][B] == 1) {
			for (i = scanline[0][E] << rshift; i > 0; i--) {
				memcpy(&scanline[0][0], &scanline[-1][0], 4);
				scanline++;
				len--;
			}
			rshift += 8;
		}
		else {
			scanline++;
			len--;
			rshift = 0;
		}
	}
	return true;
}


#endif // HDR_LOADER_H


This implements Adaptive Regularized RGF filter for 8 bit .png images.
It uses CPU and GPU using .cu files.

Compile and test:
	> sh test.sh
This will create build and test-output directories to compile and test the program.

Compile only:
	> mkdir build
	> cd build
	> cmake ../
	> make
This compiles and creates ./arrgf.out executable in build/ directory

Run program
	> ./arrgf.out <input image without .png extension> <output image without .png extension> -<parameter> <parameter details>

Example
	> ./arrgf.out -input barbara -output -test-output/barbara -arrgf -ss 4.0 -sr 0.03 -print-it 5 and 10

More info:
	> ./arrgf.out -help
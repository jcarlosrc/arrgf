# Test basic samples to see if program works.
mkdir build
cd build
cmake ../
make
cd ../
mkdir test-output
./build/arrgf.out -input barbara -output -test-output/barbara -rgf -ss 4.0 -sr 0.03 -print-it 5 and 10 -print-diff-single
./build/arrgf.out -input barbara -output test-output/barbara -arrgf -ss 4.0  -sr 0.03 -print-it 5 and 10 -print-diff-single -print-lm

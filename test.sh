# Test basic samples to see if program works.
mkdir build
cd build
cmake ../
make
cd ../
mkdir test-output
./build/arrgf.out -input barbara -output test-output/barbara -rgf -ss 4.0 -sr 0.03 -print-it 5 and 10 -print-diff-single
./build/arrgf.out -input barbara -output test-output/barbara -arrgf -ss 4.0  -sr 0.03 -print-it 5 and 10 -print-diff-single -print-lm
./build/arrgf.out -input memorial -otuput test-output/memorial -hdr-input -tone-mapping -arrgf -ss 4.0 -sr 0.1 -print-it 10 -print-input-txt -fixed-sr -print-gamma -print-txt -print-gamma 

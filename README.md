# CudaImageProcessing
Small project to experiment with filters using Cuda

There is an ImageProcessing.cu file which contains the code. This needs to be compiled using cuda's nvcc compiler (which makes use of MSVC cl compiler for the C++ side). 
This code makes use of the stb library for loading and storing png's: https://github.com/nothings/stb

The following filters are available: sepia, hblur (horizontal blur), vblur (vertical blur) and grayscale
To run the sepia filter on a png, run the following command:

ImageProcessing.exe sepia someImage.png
Where ImageProcessing.exe is the compiled code, sepia is the filter name and someImage.png is the path to png file. The processed image will appear in the directory the command is run from.

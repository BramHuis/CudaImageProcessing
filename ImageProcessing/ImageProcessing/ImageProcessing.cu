#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>

__global__ void generateSepiaImage(float *inputImage, float *outputImage, int nCols, int nRows, int nChannels) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= nCols || y >= nRows) return;

    int pixelIdx = (y * nCols + x) * nChannels;
    float pixelValueR = inputImage[pixelIdx];
    float pixelValueG = inputImage[pixelIdx + 1];
    float pixelValueB = inputImage[pixelIdx + 2];

    float outR = pixelValueR * 0.393f + pixelValueG * 0.769f + pixelValueB * 0.189f;
    float outG = pixelValueR * 0.349f + pixelValueG * 0.686f + pixelValueB * 0.168f;
    float outB = pixelValueR * 0.272f + pixelValueG * 0.534f + pixelValueB * 0.131f;

    outputImage[pixelIdx] = fminf(fmaxf(outR, 0.0f), 255.0f);
    outputImage[pixelIdx + 1] = fminf(fmaxf(outG, 0.0f), 255.0f);
    outputImage[pixelIdx + 2] = fminf(fmaxf(outB, 0.0f), 255.0f);
}

__global__ void generateGrayscaleImage(float *inputImage, float *outputImage, int nCols, int nRows, int nChannels) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= nCols || y >= nRows) return;

    int pixelIdx = (y * nCols + x) * nChannels;
    float pixelValueR = inputImage[pixelIdx];
    float pixelValueG = inputImage[pixelIdx + 1];
    float pixelValueB = inputImage[pixelIdx + 2];

    float grayscaleValue = (pixelValueR + pixelValueG + pixelValueB) / 3.0;

    outputImage[pixelIdx] = grayscaleValue;
    outputImage[pixelIdx + 1] = grayscaleValue;
    outputImage[pixelIdx + 2] = grayscaleValue;
}

__global__ void horizontalBlur(float *inputImage, float *outputImage, int nRows, int nCols, int nChannels, int blurRadius) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= nCols || y >= nRows) return;

    float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
    int totalValues = 0;

    for (int i = -blurRadius; i <= blurRadius; i++) {
        int xi = x + i;
        if (xi >= 0 && xi < nCols) {
            int neighborIdx = (y * nCols + xi) * nChannels;
            rSum += inputImage[neighborIdx];
            gSum += inputImage[neighborIdx + 1];
            bSum += inputImage[neighborIdx + 2];
            totalValues++;
        }
    }

    int pixelIdx = (y * nCols + x) * nChannels;
    outputImage[pixelIdx] = rSum / totalValues;
    outputImage[pixelIdx + 1] = gSum / totalValues;
    outputImage[pixelIdx + 2] = bSum / totalValues;
}

__global__ void verticalBlur(float *inputImage, float *outputImage, int nRows, int nCols, int nChannels, int blurRadius) {
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= nCols || y >= nRows) return;

    float rSum = 0.0f, gSum = 0.0f, bSum = 0.0f;
    int totalValues = 0;

    for (int i = -blurRadius; i <= blurRadius; i++) {
        int yi = y + i;
        if (yi >= 0 && yi < nRows) {
            int neighborIdx = (yi * nCols + x) * nChannels;
            rSum += inputImage[neighborIdx];
            gSum += inputImage[neighborIdx + 1];
            bSum += inputImage[neighborIdx + 2];
            totalValues++;
        }
    }

    int pixelIdx = (y * nCols + x) * nChannels;
    outputImage[pixelIdx] = rSum / totalValues;
    outputImage[pixelIdx + 1] = gSum / totalValues;
    outputImage[pixelIdx + 2] = bSum / totalValues;
}

bool checkInputArgs(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filter: sepia|blur> <image_file>" << std::endl;
        return false;
    }
    return true;
}

bool checkImage(unsigned char* imgDataChar, int nChannels) {
    if (imgDataChar == nullptr) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return false;
    }
    else if (nChannels != 3 && nChannels != 4) {
        std::cout << "Sepia filter requires 3 or 4 channels, this image only has " << nChannels << " channels" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (!checkInputArgs(argc, argv)) {return 1;}

    std::string filter = argv[1];
    const char* imagePath = argv[2];

    // Load the image
    int width, height, nChannels;
    unsigned char* imgDataChar = stbi_load(imagePath, &width, &height, &nChannels, 0);
    if (!checkImage(imgDataChar, nChannels)) {return 1;}
    
    
    // Calculate the number of elements 
    size_t nElements = width * height * nChannels;

    // Allocate memory for the image data in int format
    float *hImgData = new float[nElements];

    // Convert the data from char to float
    for (size_t i = 0; i < nElements; i++) {
        hImgData[i] = static_cast<float>(imgDataChar[i]);
    }
    

    float *dImgDataIn, *dImgDataOut;
    size_t nBytes = nElements * sizeof(float);
    cudaMalloc((void **)&dImgDataIn, nBytes);
    cudaMalloc((void **)&dImgDataOut, nBytes);
    cudaMemcpy(dImgDataIn, hImgData, nBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Call the appropriate kernel
    if (filter == "sepia") {
        generateSepiaImage<<<grid, block>>>(dImgDataIn, dImgDataOut, width, height, nChannels);
    } else if (filter == "hblur") {
        int blurLength = 10;
        horizontalBlur<<<grid, block>>>(dImgDataIn, dImgDataOut, width, height, nChannels, blurLength);
    } else if (filter == "vblur") {
        int blurLength = 10;
        verticalBlur<<<grid, block>>>(dImgDataIn, dImgDataOut, width, height, nChannels, blurLength);
    } else if (filter == "grayscale") {
        generateGrayscaleImage<<<grid, block>>>(dImgDataIn, dImgDataOut, width, height, nChannels);
    } else {
        std::cerr << "Unknown filter: " << filter << ". Use 'sepia' or 'blur'." << std::endl;
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(hImgData, dImgDataOut, nBytes, cudaMemcpyDeviceToHost);


    for (size_t i = 0; i < nElements; i++) {
        imgDataChar[i] = static_cast<unsigned char>(hImgData[i]); 
    }

    // Save with different filename depending on filter
    std::string outputFilename = filter + "Image.png";
    if (!stbi_write_png(outputFilename.c_str(), width, height, nChannels, imgDataChar, width * nChannels)) {
        std::cerr << "Error saving image!" << std::endl;
    }

    
    stbi_image_free(imgDataChar);
    delete [] hImgData;
    cudaFree(dImgDataIn);
    cudaFree(dImgDataOut);
    cudaDeviceReset();

    return 0;
}
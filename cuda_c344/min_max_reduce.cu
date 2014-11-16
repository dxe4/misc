#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <time.h>

// cuda-memcheck ./a.out debug segfault
// nvcc -arch=sm_21 min_max_reduce.cu -G0 # compile
#define H_MIN(a,b) (((a)<(b))?(a):(b))
#define H_MAX(a,b) (((a)>(b))?(a):(b))


typedef float (*reduce_cb) (float &, float &);

__device__ float MIN(float &x, float &y)
{
    return x < y ? x : y;
}

__device__ float MAX(float &x, float &y)
{
    return x > y ? x : y;
}

template<reduce_cb cb>
__global__ void reduce(
    float *input, float *output, int *n, int *nRows, int *nCols, int *blocksY)
{
    /**
    __shared__ temp has a max size of 49152 b
    so the blokcs are split accordingly
    So now we need x values writen in outpu where x=blockSize.x * blockSize.y
    then we need to launch another kernel with input arr[x]
    Every block executes 1 blelloch
    **/
    extern __shared__ float temp[];// allocated on invocation
    __shared__ float last_elm;

    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);

    const int thid = thread_2D_pos.y * ( *nCols ) + thread_2D_pos.x;
    const int b_thid = threadIdx.x;
    int offset = 1;

    if (2 * thid + 1 >= *n)
    {
        return;
    }
    temp[b_thid] = input[2 * thid]; // load input into shared memory
    temp[b_thid + 1] = input[2 * thid + 1];

    __syncthreads();
    // build sum in place up the tree
    for (int d = 1024 >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (b_thid < d)
        {
            const int ai = offset * (b_thid + 1) - 1;
            const int bi = offset * (b_thid + 2) - 1;
            temp[bi] = cb(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    // clear the last element
    if (b_thid == 0)
    {
        last_elm = temp[1023];
        temp[1023] = 0.f;
    }
    //offset = *n;
    // traverse down tree & build scan
    __syncthreads();
    for (int d = 1; d < 1024; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (b_thid < d)
        {
            int ai = offset * (b_thid + 1) - 1;
            int bi = offset * (b_thid + 2) - 1;

            if (ai >= 0 && bi >= 0)
            {
                float swap_temp = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] = cb(swap_temp, temp[bi]);
            }
        }
    }
    __syncthreads();
    // write results to device memory
    if (b_thid == 0)
    {
        int index = (blockIdx.x * (*blocksY ) + blockIdx.y);
        output[index] = cb(temp[1023], last_elm);
    }
}


int shared_memory_per_block()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    int sharedMemPerBlock = -1;

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        sharedMemPerBlock = prop.sharedMemPerBlock;
    }
    return sharedMemPerBlock;
}

int main(int argc, char **argv)
{
    int numCols = 1536;
    int numRows = 1024;

    int ARRAY_SIZE = numCols * numRows;
    int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    time_t rand_t;
    srand((unsigned) time(&rand_t));

    float *h_in =  (float *)malloc(ARRAY_BYTES);
    float *h_out = (float *)malloc(ARRAY_BYTES + sizeof(float));

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = rand() % 20000;
    }


    float *d_in;
    float *d_out;
    int _h_in = ARRAY_SIZE;
    int *h_n = &_h_in;
    int *d_n, *nRows, *nCols, *blocksY;

    // this is the amount of __shared__ we can use
    int sharedMemPerBlock = shared_memory_per_block();
    // Split kernels to match the cache size in a square
    int blocks = (int)floor(sqrt(ARRAY_BYTES / sharedMemPerBlock));

    checkCudaErrors(cudaMalloc((void **) &d_in, ARRAY_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_out, ARRAY_BYTES + sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_n, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &nCols, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &nRows, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &blocksY, sizeof(int)));

    checkCudaErrors(cudaMemcpy(
        d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_n, h_n, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        nRows, &numRows, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        nCols, &numCols, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        blocksY, &blocks, sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(blocks, blocks, 1);
    // good luck here +++
    reduce<MAX> <<< blockSize, 1024, 1024 * sizeof(float)>>>(
        d_in, d_out, d_n, nRows, nCols, blocksY);
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy(
        h_out, d_out, ARRAY_BYTES + sizeof(float), cudaMemcpyDeviceToHost));

    // cudaMemcpyFromSymbol(&h_output, "d_output", sizeof(float), 0, cudaMemcpyDeviceToHost);

    float max = -1;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (h_in[i] > max)
        {
            max = h_in[i];
        }
    }
    printf("%f max\n", max);
    printf("%f last elm\n", h_out[ARRAY_SIZE]);

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_n));
    checkCudaErrors(cudaFree(d_out));

    return 0;
}
//http://blog.codinghorror.com/content/images/uploads/2008/08/6a0120a85dcdae970b012877705d12970c-pi.jpg

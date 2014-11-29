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
    float *input, float *output, int *n, int *n_rows, int *n_cols,
    int *d_output_size)
{
    /**
    __shared__ temp has a max size of 49152 b
    so the blokcs are split accordingly
    So now we need x values writen in outpu where x=blockSize.x * blockSize.y
    then we need to launch another kernel with input arr[x]
    Every block executes 1 blelloch
    **/
    extern __shared__ float temp[]; // allocated on invocation
    __shared__ float last_elm;

    const int2 thread_2D_pos = make_int2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thid = thread_2D_pos.y * ( *n_cols ) + thread_2D_pos.x;
    const int b_thid = threadIdx.x * blockDim.y + threadIdx.y;
    int offset = 1;

    if (2 * thid + 1 >= *n)
    {
        return;
    }
    temp[b_thid] = input[2 * thid]; // load input into shared memory
    temp[b_thid + 1] = input[2 * thid + 1];

    __syncthreads();

    // build sum in place up the tree
    for (int d = *d_output_size >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (b_thid < d)
        {
            const int ai = offset * (2 * b_thid + 1) - 1;
            const int bi = offset * (2 * b_thid + 2) - 1;
            temp[bi] = cb(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    // clear the last element
    if (b_thid == 0)
    {
        last_elm = temp[*d_output_size - 1];
        temp[*d_output_size - 1] = 0.f;
    }
    // offset = *d_output_size;
    // traverse down tree & build scan
    __syncthreads();

    for (int d = 1; d < *d_output_size; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (b_thid < d)
        {
            const int ai = offset * (2 * b_thid + 1) - 1;
            const int bi = offset * (2 * b_thid + 2) - 1;
            if (ai >= 0 && bi >= 0)
            {
                float swap_temp = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] = cb(swap_temp, temp[bi]);
                temp[bi] = bi;
            }
        }
    }
    __syncthreads();

    // write results to device memory
    if (*d_output_size == blockDim.x * blockDim.y)
    {
        if (b_thid == 0)
        {
            // there may be something wrong here
            int index = (blockIdx.x * blockDim.y + blockIdx.y);
            output[index] = cb(temp[*d_output_size - 1], last_elm);
        }
    }
    else
    {
        output[b_thid] = temp[b_thid];
        if (b_thid == *d_output_size - 1)
        {
            output[*d_output_size - 1] = cb(temp[*d_output_size - 1], last_elm);
        }
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
    int numCols = 1516;
    int numRows = 1024;
    int h_output_size = 32 * 32;

    int ARRAY_SIZE = numCols * numRows;
    int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    time_t rand_t;
    srand((unsigned) time(&rand_t));

    float *h_in =  (float *)malloc(ARRAY_BYTES);
    float *h_out = (float *)malloc(ARRAY_BYTES + sizeof(float));

    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = rand() % 200000;
    }

    float *d_in;
    float *d_out;
    int _h_in = ARRAY_SIZE;
    int *h_n = &_h_in;
    int *d_n, *n_rows, *n_cols, *d_output_size;

    // this is the amount of __shared__ we can use
    int sharedMemPerBlock = shared_memory_per_block();
    // Split kernels to match the cache size in a square
    int blocks = ARRAY_SIZE / 1024; // (int)floor(sqrt(ARRAY_SIZE / 1024)) + 1;

    checkCudaErrors(cudaMalloc((void **) &d_in, ARRAY_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_out, ARRAY_BYTES + sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_n, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &n_cols, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &n_rows, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_output_size, sizeof(int)));

    checkCudaErrors(cudaMemcpy(
        d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_n, h_n, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        n_rows, &numRows, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        n_cols, &numCols, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_output_size, &h_output_size, sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32, 1);
    dim3 gridSize(numRows / 32, numCols / 32, 1);
    int shared_memoery_size = (numRows / 31 * numCols / 31) * sizeof(float);
    // good luck here +++
    reduce<MAX> <<< gridSize, blockSize, shared_memoery_size>>>( 
        d_in, d_out, d_n, n_rows, n_cols, d_output_size);
    cudaThreadSynchronize();
    cudaDeviceSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy(
        h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    cudaFree((void **) &d_in);
    cudaFree((void **) &d_out);
    cudaFree((void **) &d_output_size);

    float max_a = -1;
    float max = -1;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (h_in[i] > max)
        {
            max = h_in[i];
        }
    }
    // temp hack
    for (int i = 0; i < 150 + (32 * 32); i++)
    {
        if (h_out[i] > max_a)
        {
            max_a = h_out[i];
        }
        printf("%f %i --\n", h_out[i], i);
    }
    // occasionally we get a 199998 instead of 199999
    printf("%f %f\n", max_a, max);
    h_output_size = blocks * blocks;
    _h_in = blocks * blocks;

    //free(h_out);
    //h_out = (float *) malloc(ARRAY_BYTES + sizeof(float));

    checkCudaErrors(cudaMemcpy(
        d_output_size, &h_output_size, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_in, h_out, ARRAY_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_n, &_h_in, sizeof(int), cudaMemcpyHostToDevice));

    reduce<MAX> <<<  h_output_size, 1, blocks * blocks * sizeof(float)>>>(
        d_in, d_out, d_n, n_rows, n_cols, d_output_size);

    cudaThreadSynchronize();
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(
        h_out, d_out, blocks * blocks * sizeof(float), cudaMemcpyDeviceToHost));


    // cudaMemcpyFromSymbol(&h_output, "d_output", sizeof(float), 0, cudaMemcpyDeviceToHost);

    max = -1;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (h_in[i] > max)
        {
            max = h_in[i];
        }
    }
    for (int i = 0; i < blocks * blocks; i++)
    {
        if (h_out[i] > max_a)
        {
            max_a = h_out[i];
        }
        // printf("%f %i --\n", h_out[i], i);
    }
    printf("%f %f max\n", max, max_a);
    printf("%f last elm\n", h_out[ARRAY_SIZE]);

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_n));
    checkCudaErrors(cudaFree(d_out));

    return 0;
}
//http://blog.codinghorror.com/content/images/uploads/2008/08/6a0120a85dcdae970b012877705d12970c-pi.jpg

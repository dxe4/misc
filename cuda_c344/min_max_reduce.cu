#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <time.h>

// cuda-memcheck ./a.out debug segfault
// nvcc -arch=sm_21 min_max_reduce.cu -G0 # compile


typedef float (*reduce_cb) (float &, float &);

__device__ float MIN(float &x, float &y)
{
    return x < y ? x : y;
}

__device__ float MAX(float &x, float &y)
{
    return x > y ? x : y;
}


__device__ float d_output;

template<reduce_cb cb>
__global__ void reduce(float *input, int *n)
{
    /**
    TODO
    a) avoid bank conflicts (rtfd)
    c) the shared memory size could be reduced
    **/
    extern __shared__ float temp[];// allocated on invocation

    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);
    const int thid = thread_2D_pos.y * (*n / 2) + thread_2D_pos.x;
    int offset = 1;

    if (2 * thid + 1 >= *n)
    {
        return;
    }
    temp[2 * thid] = input[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = input[2 * thid + 1];

    __syncthreads();

    // build sum in place up the tree
    for (int d = *n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] = cb(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    // clear the last element
    if (thid == 0)
    {
        temp[*n - 1] = 0.f;
    }
    offset = *n;
    // traverse down tree & build scan

    for (int d = 1; d < *n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            if (ai >= 0 && bi >= 0)
            {
                float t = temp[ai];
                temp[ai] = temp[bi];
                // temp[bi] += t;
                temp[bi] = cb(t, temp[bi]);
            }
        }
    }
    __syncthreads();
    if(2 * thid + 1 == *n-1) {
        /**
        Note, normally we would need this
            -    output[2 * thid] = temp[2 * thid]; // write results to device memory
            -    output[2 * thid + 1] = temp[2 * thid + 1];
        But because we only need the last value we pick it from the appropriate thread
        **/
        d_output = temp[*n-1];
    }
}


int main(int argc, char **argv)
{
    // TODO picking grid/block and using it right needs to be fixed
    const int ARRAY_SIZE = 1024 * 2;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    time_t rand_t;
    srand((unsigned) time(&rand_t));

    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = (rand() % 2000 + rand() % 2000);
    }
    // float h_out[ARRAY_SIZE];
    float h_output = 0;

    float *d_in;
    // float *d_out;
    int _h_in = ARRAY_SIZE;
    int *h_n = &_h_in;
    int *d_n;

    checkCudaErrors(cudaMalloc((void **) &d_in, ARRAY_BYTES));
    // checkCudaErrors(cudaMalloc((void **) &d_out, ARRAY_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_n, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockSize(24, 24, 1);
    reduce<MAX> <<<blockSize, 1024, ARRAY_SIZE *sizeof(float)>>>(d_in, d_n);
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // checkCudaErrors(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    cudaMemcpyFromSymbol(&h_output, "d_output", sizeof(float), 0, cudaMemcpyDeviceToHost);

    printf("%f\n", h_output);


    float max = -1;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (h_in[i] > max)
        {
            max = h_in[i];
        }
    }
    printf("%f max\n", max);
    // printf("%f last elm\n", h_out[ARRAY_SIZE - 1]);

    checkCudaErrors(cudaFree(d_in));

    checkCudaErrors(cudaFree(d_n));

    return 0;
}

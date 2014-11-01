#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

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
__global__ void reduce(float *input, float *output, int *n)
{
    extern __shared__ float temp[];// allocated on invocation

    int thid = threadIdx.x;
    int offset = 1;

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
            printf("%i %i\n", ai, bi);
            temp[bi] = cb(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    __syncthreads();
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
    output[2 * thid] = temp[2 * thid]; // write results to device memory
    output[2 * thid + 1] = temp[2 * thid + 1];
}


int main(int argc, char **argv)
{
    // TODO picking grid/block and using it right needs to be fixed
    const int ARRAY_SIZE = 16;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    time_t t;
    srand((unsigned) time(&t));

    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = rand() % 2000;
    }
    float h_out[ARRAY_SIZE];

    float *d_in;
    float *d_out;
    int _h_in = ARRAY_SIZE;
    int *h_n = &_h_in;
    int *d_n;

    checkCudaErrors(cudaMalloc((void **) &d_in, ARRAY_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_out, ARRAY_BYTES));
    checkCudaErrors(cudaMalloc((void **) &d_n, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice));

    reduce<MAX> <<< 1, ARRAY_SIZE, ARRAY_SIZE *sizeof(float)>>>(d_in, d_out, d_n);


    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    float max = -1;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%f, ", h_in[i]);
        if(h_in[i] > max){
            max = h_in[i];
        }   
    }
    printf("\n");
    printf("\n");
    printf("%f max\n", max);
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%f, ", h_out[i]);
    }

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return 0;
}

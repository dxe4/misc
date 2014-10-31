/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

#include <stdio.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


/*
For Bank Conflicts:
https://www.youtube.com/watch?v=CZgM3DEBplE
http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

Reduce:
for d = 0 to log2 n – 1 do
     for all k = 0 to n – 1 by 2 d+1 in parallel do
          x[k +  2 d+1 – 1] = x[k +  2 d  – 1] + x[k +  2 d +1 – 1]

Down sweep:
x[n – 1] = 0
for d = log2 n – 1 down to 0 do
      for all k = 0 to n – 1 by 2 d +1 in parallel do
           t = x[k +  2 d  – 1]
           x[k +  2 d  – 1] = x[k +  2 d +1 – 1]
           x[k +  2 d +1 – 1] = t +  x[k +  2 d +1 – 1]
*/

__global__ void find_minmax(const float *const input, float *d_output, int n)
{
    extern __shared__ float temp[];  // allocated on invocation
    int t_id = threadIdx.x;
    int offset = 1;

    int ai = t_id;
    int bi = t_id + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // load input into shared memory
    temp[ai + bankOffsetA] = input[ai];
    temp[bi + bankOffsetB] = input[bi];

    // build sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (t_id < d)
        {
            int ai = offset * (2 * t_id + 1) - 1;
            int bi = offset * (2 * t_id + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] = MAX(temp[ai], temp[bi]);
        }
        offset *= 2;

        if (t_id == 0)
        {
            temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        } // clear the last element


        // traverse down tree & build scan
        for (int d = 1; d < n; d *= 2)
        {
            offset >>= 1;
            __syncthreads();
            if (t_id < d)
            {
                int ai = offset * (2 * t_id + 1) - 1;
                int bi = offset * (2 * t_id + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] = MAX(t, temp[bi]);
            }
        }
        __syncthreads();

        temp[n + ai] = temp[ai + bankOffsetA];
        temp[n + bi] = temp[bi + bankOffsetB];
    }

    __syncthreads();
    d_output = &temp[n * 2];
}


void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    printf("%f\n", d_logLuminance[0]);
    dim3 blockSize(24, 24, 1);
    dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);


    float *d_output;
    int *d_size;
    int _h_size = numCols * numRows;
    int *h_size = &_h_size;
    checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_size, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_size, h_size, sizeof(int), cudaMemcpyHostToDevice));

    find_minmax<<<gridSize, blockSize, numCols * numRows * 2 * sizeof(float)>>>(d_logLuminance, d_output, *d_size);
    //TODO
    /*Here are the steps you need to implement
      1) find the minimum and maximum value in the input logLuminance channel
         store in min_logLum and max_logLum
      2) subtract them to find the range
      3) generate a histogram of all the values in the logLuminance channel using
         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
      4) Perform an exclusive scan (prefix sum) on the histogram to get
         the cumulative distribution of luminance values (this should go in the
         incoming d_cdf pointer which already has been allocated for you)       */
}

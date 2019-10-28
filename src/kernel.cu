#include "libs/helper_cuda.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
	return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

__global__ void randInit(curandState* state, uint64_t seed)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, tid, 0, &state[tid]);
}

template <typename T>
__device__ bool inMainCardioid(T real, T imag)
{
	T imag_squared = imag * imag;
	T q = real - T(0.25);
	q = q * q + imag_squared;
	return q * (q + (real - T(0.25))) < (imag_squared * T(0.25));
}

template <typename T>
__device__ bool inOrder2Bulb(T real, T imag)
{
	++real;
	real = real * real;
	return (real + imag * imag) < T(1.0 / 16);
}

template<class T>
__device__ inline uint32_t CalcMandelbrot(T xPos, T yPos, uint32_t maxIter)
{
	T x = 0, y = 0, xx = 0, yy = 0;
	uint32_t i = 0;
	while (i++ < maxIter && (xx + yy < T(4.0)))
	{
		y = x * y * T(2.0) + yPos;
		x = xx - yy + xPos;
		yy = y * y;
		xx = x * x;
	}

	return i;
}

template<typename T, unsigned numChannels = 4>
__global__ void
__launch_bounds__(1024) // reduce # used registers
Buddhabrot(uint32_t* dst, int imageW, int imageH, uint32_t maxIter,
	T xOff, T yOff, T scale, curandState* randStates)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	curandState localState = randStates[tid];
	for (uint32_t i = 0; i < 64; ++i)
	{
		float x = curand_uniform(&localState);
		float y = curand_uniform(&localState);
		const T xPos = x * imageW * scale + xOff;
		const T yPos = y * imageH * scale + yOff;

		if (inMainCardioid(xPos, yPos) || inOrder2Bulb(xPos, yPos))
			continue;

		uint32_t m = CalcMandelbrot<T>(xPos, yPos, maxIter);
		//m = blockIdx.x;         // uncomment to see scheduling order

		// we converged, skip that
		if (m < 10)
			continue;

		uint32_t channel = 0;
		if (numChannels > 1 && m > 5000)
			channel = 1;
		if (numChannels > 2 && m > 10000)
			channel = 2;

		x = 0, y = 0;
		for (int j = 0; j < m; ++j)
		{
			float yy = y * y;
			y = x * y * 2 + yPos;
			x = x * x - yy + xPos; // inline yy to get the Kleeblatt

			uint32_t ix = uint32_t((x - xOff) / scale);
			uint32_t iy = uint32_t((y - yOff) / scale);
			if (!ix || !iy || // cast negative float to uint is 0 probably, so skip those
				ix >= imageW || iy >= imageH)
				continue;
			uint32_t idx = (iy * imageW + ix) * numChannels + channel;
			atomicAdd(&dst[idx], 1);
		}
	}
	randStates[tid] = localState;
}

void RunRandInit(curandState* state, size_t len, uint64_t seed)
{
	randInit <<<len / 256, 256>>> (state, seed);
	getLastCudaError("randInit kernel execution failed.\n");
}

__global__ void computeMinMax(uint32_t* data, size_t size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t minimum = ~0u, maximum = 0;
	for (; tid < size; tid += gridDim.x * blockDim.x)
	{
		minimum = min(minimum, data[tid]);
		maximum = max(maximum, data[tid]);
	}
	atomicMin(&data[0], minimum);
	atomicMax(&data[1], maximum);
}

void RunBuddhabrot(uint32_t* dst, int imageW, int imageH, double xOff, double yOff, double scale, int numSMs, curandState* randStates)
{
	uint32_t maxIter = 8192*2;
	Buddhabrot<float> <<<16 * numSMs, 256>>> (dst, imageW, imageH, maxIter, (float)xOff, (float)yOff,
		(float)scale, randStates);

	getLastCudaError("Buddhabrot kernel execution failed.\n");

	computeMinMax <<<16 * numSMs, 1024>>> (dst, imageW * imageH);
	getLastCudaError("minmax kernel execution failed.\n");
}

#include "simulation/tableau.cuh"
#include "core/types.cuh"
#include <cuda_runtime.h>

namespace RKF78
{
    __constant__ double d_c[NStages - 1];
    __constant__ double d_b[NStages];
    __constant__ double d_be[NStages];
    __constant__ double d_a[NStages - 1][NStages - 1];

    cudaError_t initialize_device_tableau()
    {
        cudaError_t err;
        err = cudaMemcpyToSymbol(d_c, host::c, sizeof(host::c));
        if (err != cudaSuccess)
        {
            return err;
        }
        err = cudaMemcpyToSymbol(d_b, host::b, sizeof(host::b));
        if (err != cudaSuccess)
        {
            return err;
        }
        err = cudaMemcpyToSymbol(d_be, host::be, sizeof(host::be));
        if (err != cudaSuccess)
        {
            return err;
        }
        err = cudaMemcpyToSymbol(d_a, host::a, sizeof(host::a));
        return err;
    }
}
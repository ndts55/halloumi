#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace RKF78
{
    constexpr std::size_t OdeOrder = 1;
    constexpr std::size_t NStages = 13;
    constexpr std::size_t Order = 8;
    constexpr bool IsEmbedded = true; // not actually necessary

    namespace host
    {
        // Nodes for the stages, used in the main method, omits the first value
        static const double c[NStages - 1] = {2. / 27., 1. / 9., 1. / 6., 5. / 12., 1. / 2., 5. / 6., 1. / 6., 2. / 3., 1. / 3., 1., 0., 1.};
        // Weights for the stages, for the main method
        static const double b[NStages] = {0., 0., 0., 0., 0., 34. / 105., 9. / 35., 9. / 35., 9. / 280., 9. / 280., 0., 41. / 840., 41. / 840.};
        // Embedded weights, for the embedded method
        static const double be[NStages] = {41. / 840., 0., 0., 0., 0., 0., 0., 0., 0., 0., 41. / 840., -41. / 840., -41. / 840.};
        // Coefficient matrix, leaves out first row, and last column, which are always 0
        static const double a[NStages - 1][NStages - 1] = {
            {2. / 27., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
            {1. / 36., 1. / 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
            {1. / 24., 0., 1. / 8., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
            {5. / 12., 0., -25. / 16., 25. / 16., 0., 0., 0., 0., 0., 0., 0., 0.},
            {1. / 20., 0., 0., 1. / 4., 1. / 5., 0., 0., 0., 0., 0., 0., 0.},
            {-25. / 108., 0., 0., 125. / 108., -65. / 27., 125. / 54., 0., 0., 0., 0., 0., 0.},
            {31. / 300., 0., 0., 0., 61. / 225., -2. / 9., 13. / 900., 0., 0., 0., 0., 0.},
            {2., 0., 0., -53. / 6., 704. / 45., -107. / 9., 67. / 90., 3., 0., 0., 0., 0.},
            {-91. / 108., 0., 0., 23. / 108., -976. / 135., 311. / 54., -19. / 60., 17. / 6., -1. / 12., 0., 0., 0.},
            {2383. / 4100., 0., 0., -341. / 164., 4496. / 1025., -301. / 82., 2133. / 4100., 45. / 82., 45. / 164., 18. / 41., 0., 0.},
            {3. / 205., 0., 0., 0., 0., -6. / 41., -3. / 205., -3. / 41., 3. / 41., 6. / 41., 0., 0.},
            {-1777. / 4100., 0., 0., -341. / 164., 4496. / 1025., -289. / 82., 2193. / 4100., 51. / 82., 33. / 164., 12. / 41., 0., 1.}};
    }

    extern __constant__ double d_c[NStages - 1];
    extern __constant__ double d_b[NStages];
    extern __constant__ double d_be[NStages];
    extern __constant__ double d_a[NStages - 1][NStages - 1];

    __device__ inline double coefficient(std::size_t i, std::size_t j)
    {
        if (i == 0 || j == (NStages - 1))
        {
            return 0.0;
        }

        return d_a[i][j];
    }

    __device__ inline double weight(std::size_t i) { return d_b[i]; }

    __device__ inline double embedded_weight(std::size_t i) { return d_be[i]; }

    __device__ inline double node(std::size_t i)
    {
        if (i == 0)
        {
            return 0.0;
        }
        return d_c[i];
    }

    cudaError_t initialize_device_tableau();
}

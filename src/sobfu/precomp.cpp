/* sobfu includes */
#include <sobfu/precomp.hpp>

/* checks if x is a power of 2 */
bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

/* computes the nearest power of 2 larger than x */
int nextPow2(int x) {
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

void get_num_blocks_and_threads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads) {
    /* get device capability, to avoid block/grid size exceed the upper bound */
    cudaDeviceProp prop;
    int device;
    cudaSafeCall(cudaGetDevice(&device));
    cudaSafeCall(cudaGetDeviceProperties(&prop, device));

    threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
    blocks  = (n + (threads * 2 - 1)) / (threads * 2);

    if ((float) threads * blocks > (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0]) {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n", blocks,
               prop.maxGridSize[0], threads * 2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = std::min(maxBlocks, blocks);
}

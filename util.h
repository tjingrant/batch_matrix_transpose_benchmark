#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_AXIS_KERNEL_LOOP(i, n, axis)                                  \
  for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n.axis; \
       i += blockDim.axis * gridDim.axis)

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

struct CudaLaunchConfig {
  // Logical number of thread that works on the elements. If each logical
  // thread works on exactly a single element, this is the same as the working
  // element count.
  int virtual_thread_count = -1;
  // Number of threads per block.
  int thread_per_block = -1;
  // Number of blocks for Cuda kernel launch.
  int block_count = -1;
};

// Calculate the Cuda launch config we should use for a kernel launch.
// This is assuming the kernel is quite simple and will largely be
// memory-limited.
inline CudaLaunchConfig GetCudaLaunchConfig(int work_element_count) {
  int maxCudaThreadsPerMultiProcessor = 1024;
  int numCudaMultiProcessors = 15;
  CudaLaunchConfig config;

  // in case of invalid input, return the default value config, which has all -1
  if (work_element_count <= 0) {
    return config;
  }

  const int virtual_thread_count = work_element_count;
  const int physical_thread_count = std::min(
      numCudaMultiProcessors * maxCudaThreadsPerMultiProcessor,
      virtual_thread_count);
  const int thread_per_block = std::min(1024, maxCudaThreadsPerMultiProcessor);
  const int block_count =
      std::min(DIV_UP(physical_thread_count, thread_per_block),
               numCudaMultiProcessors);

  config.virtual_thread_count = virtual_thread_count;
  config.thread_per_block = thread_per_block;
  config.block_count = block_count;
  return config;
}
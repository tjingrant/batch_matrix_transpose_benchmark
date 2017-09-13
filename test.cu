
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "util.h"

#define EIGEN_DEVICE_FUNC __host__ __device__
#define EIGEN_STRONG_INLINE __forceinline__

// TODO(mjanusz): Move this to a shared util file.
// A simple array that contains data that can be passed between CPU and GPU.
template <typename T, int IndexCount, T DefaultValue>
struct Array {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator[](int index) const {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T& operator[](int index) {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array() {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0) {
    data[0] = a0;
    for (int i = 1; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0, T a1) {
    data[0] = a0;
    data[1] = a1;
    for (int i = 2; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Array(T a0, T a1, T a2) {
    data[0] = a0;
    data[1] = a1;
    data[2] = a2;
    for (int i = 3; i < IndexCount; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_STRONG_INLINE Array(const std::array<T, IndexCount>& array) {
    for (int i = 0; i < IndexCount; i++) {
      data[i] = array[i];
    }
  }
  T data[IndexCount];
};

// A dimension type with compile-time known size.
template <int IndexCount>
struct Dimension : Array<int, IndexCount, 1> {
  typedef Array<int, IndexCount, 1> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0) : Base(a0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0, int a1)
      : Base(a0, a1) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dimension(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
  EIGEN_STRONG_INLINE Dimension(const std::array<int, IndexCount>& array)
      : Base(array) {}
};

// An index type with compile-time known size.
template <int IndexCount>
struct Index : Array<int, IndexCount, 0> {
  typedef Array<int, IndexCount, 0> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index(int a0) : Base(a0) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index(int a0, int a1) : Base(a0, a1) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
};

// A helper function that converts a tensor index into a flat array index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int TensorIndexToFlat(
    const Index<IndexCount>& index, const Dimension<IndexCount>& dims) {
  int flat_index = index[0];
  for (int i = 1; i < IndexCount; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// A helper function that converts a flat array index into a tensor index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index<IndexCount> FlatToTensorIndex(
    int index, const Dimension<IndexCount>& dims) {
  Index<IndexCount> tensor_index;
  for (int i = IndexCount - 1; i >= 0; i--) {
    tensor_index[i] = index % dims[i];
    index /= dims[i];
  }
  return tensor_index;
}

// A Cuda custom kernel that swaps dimension-0 and dimension-2 of a 3D tensor.
template <typename T>
__global__ void SwapDimension0And2InTensor3Simple(int nthreads, const T* input,
                                                  Dimension<3> input_dims,
                                                  T* output) {
  Dimension<3> output_dims;
  output_dims[0] = input_dims[2];
  output_dims[1] = input_dims[1];
  output_dims[2] = input_dims[0];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;

    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[2];
    input_tensor_index[1] = output_tensor_index[1];
    input_tensor_index[2] = output_tensor_index[0];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = __ldg(input + input_index);
  }
}

// A Cuda custom kernel that swaps dimension-1 and dimension-2 of a 3D tensor.
template <typename T>
__global__ void SwapDimension1And2InTensor3Simple(int nthreads, const T* input,
                                                  Dimension<3> input_dims,
                                                  T* output) {
  Dimension<3> output_dims;
  output_dims[0] = input_dims[0];
  output_dims[1] = input_dims[2];
  output_dims[2] = input_dims[1];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int output_index = index;
    Index<3> output_tensor_index = FlatToTensorIndex(output_index, output_dims);

    Index<3> input_tensor_index;
    input_tensor_index[0] = output_tensor_index[0];
    input_tensor_index[1] = output_tensor_index[2];
    input_tensor_index[2] = output_tensor_index[1];

    int input_index = TensorIndexToFlat(input_tensor_index, input_dims);

    output[output_index] = __ldg(input + input_index);
  }
}

// Use shared memory tiles to swap dimension-1 and dimension-2 of a 3D tensor,
// where dimensions are zero-based: output[i][j][k] = input[i][k][j].
//
// Each thread block operates on a single tile, a square of dimensions TileSize
// x TileSize.  We require that the thread block's X dimension equals TileSize,
// and its Y dimension equals NumSubTiles.
//
// For best performance, you should probably set TileSize equal to the number of
// threads in a warp (32 in nvidia GPUs).  With a TileSize of 32, NumSubTiles ==
// 4 or 8 seems to get the best performance on K40 GPUs.
template <typename T, int TileSize, int NumSubTiles>
__global__ void SwapDimension1And2InTensor3UsingTiles(const T* input,
                                                      Dimension<3> input_dims,
                                                      T* output) {
  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ T shared_memory_tile[TileSize][TileSize + 1];

  static_assert(TileSize % NumSubTiles == 0,
                "TileSize must be divisible by NumSubTiles");


//#define MY_ASSERT(X) if (!(X)) printf(#X)
#define MY_ASSERT(X)
  MY_ASSERT(blockDim.x == TileSize);
  MY_ASSERT(blockDim.y == NumSubTiles);
  MY_ASSERT(blockDim.z == 1);
  MY_ASSERT(gridDim.y == 1);
  MY_ASSERT(gridDim.z == 1);

  // We break down the tile into NumSubTiles groups, so each thread processes
  // kSubTileSize elements (except at the edges of the input).
  const int kSubTileSize = TileSize / NumSubTiles;

  int x = threadIdx.x;

  Dimension<3> output_dims = {
      input_dims[0], input_dims[2], input_dims[1],
  };

  Dimension<3> input_dims_in_tiles = {
      input_dims[0], (input_dims[1] + TileSize - 1) / TileSize,
      (input_dims[2] + TileSize - 1) / TileSize,
  };

  Index<3> input_tile_index =
      FlatToTensorIndex(blockIdx.x, input_dims_in_tiles);

  Index<3> input_tile_origin = {
      input_tile_index[0], input_tile_index[1] * TileSize,
      input_tile_index[2] * TileSize,
  };

  int input_origin_flat_index =
      TensorIndexToFlat(input_tile_origin, input_dims);

  int tile_width = TileSize;
  // Only the last row or column may not have the full size.
  if (input_tile_index[2] == input_dims_in_tiles[2] - 1) {
    tile_width = input_dims[2] - (input_dims_in_tiles[2] - 1) * TileSize;
  }
  int tile_height = TileSize;
  if (input_tile_index[1] == input_dims_in_tiles[1] - 1) {
    tile_height = input_dims[1] - (input_dims_in_tiles[1] - 1) * TileSize;
  }

  int input_flat_index = input_origin_flat_index + x;
  int y_start = static_cast<int>(threadIdx.y) * kSubTileSize;

  // Load the data from input memory to the shared memory tile.
  if (x < tile_width) {
    int y_end = min(y_start + kSubTileSize, tile_height);
    for (int y = y_start; y < y_end; y++) {
      shared_memory_tile[y][x] = input[input_flat_index + y * input_dims[2]];
    }
  }

  __syncthreads();

  Index<3> output_tile_index = {
      input_tile_index[0], input_tile_index[2], input_tile_index[1],
  };

  Index<3> output_tile_origin = {
      output_tile_index[0], output_tile_index[1] * TileSize,
      output_tile_index[2] * TileSize,
  };

  int output_origin_flat_index =
      TensorIndexToFlat(output_tile_origin, output_dims);

  int output_flat_index = output_origin_flat_index + x;

  // Load the data from the shared memory tile to the output memory.
  if (x < tile_height) {
    int y_end = min(y_start + kSubTileSize, tile_width);
    for (int y = y_start; y < y_end; y++) {
      output[output_flat_index + y * output_dims[2]] = shared_memory_tile[x][y];
    }
  }
}

#define THREAD_NUM 1024
#define TILE_SIZE_I 512
#define TILE_SIZE_J 8
#define PADDING 1

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
  #define T float
  #define READ_ROW_PER_PASS (THREAD_NUM/TILE_SIZE_J)
  #define WRITE_ROW_PER_PASS (THREAD_NUM/TILE_SIZE_I)

__global__ void MySwapDimension1And2InTensor3UsingTiles(const T* __restrict__ input,
                                        Dimension<3> input_dims,
                                        T* __restrict__ output) {

  // One extra line in the inner dimension to avoid share memory bank conflict.
  __shared__ T shared_memory_tile[TILE_SIZE_I][TILE_SIZE_J+PADDING];

  #define SHARED(i, j)\
    shared_memory_tile[i][j]

  int x = threadIdx.x;

  Dimension<3> output_dims = {
      input_dims[0], input_dims[2], input_dims[1],
  };

  Dimension<3> input_dims_in_tiles = {
      input_dims[0], (input_dims[1] + (TILE_SIZE_I) - 1) / (TILE_SIZE_I),
      (input_dims[2] + (TILE_SIZE_J) - 1) / (TILE_SIZE_J),
  };

  Index<3> input_tile_index =
      FlatToTensorIndex(blockIdx.x, input_dims_in_tiles);

  Index<3> input_tile_origin = {
      input_tile_index[0], input_tile_index[1] * (TILE_SIZE_I),
      input_tile_index[2] * (TILE_SIZE_J),
  };

  int input_origin_flat_index =
      TensorIndexToFlat(input_tile_origin, input_dims);

  #define INPUT(i, j)\
      input[input_origin_flat_index +\
        (i) * input_dims[2] + (j)]

  int ti = x/TILE_SIZE_J;
  int tj = x%TILE_SIZE_J;

  for (int i_loc = ti; i_loc < (TILE_SIZE_I); i_loc += READ_ROW_PER_PASS) {
    //if (threadIdx.y > 16)
    //  printf("%d, %d\n", threadIdx.y, j);
    SHARED(i_loc, tj) = INPUT(i_loc, tj);
  }

  __syncthreads();

  Index<3> output_tile_index = {
      input_tile_index[0], input_tile_index[2], input_tile_index[1],
  };

  Index<3> output_tile_origin = {
      output_tile_index[0], output_tile_index[1] * TILE_SIZE_J,
      output_tile_index[2] * TILE_SIZE_I,
  };

  int output_origin_flat_index =
      TensorIndexToFlat(output_tile_origin, output_dims);

  // Oriented with respect to the output array.
  ti = x/TILE_SIZE_I;
  tj = x%TILE_SIZE_I;

  #define OUTPUT(i, j)\
      output[output_origin_flat_index +\
        (i) * output_dims[2] + (j)]

  for (int i_loc = ti; i_loc < (TILE_SIZE_J); i_loc += WRITE_ROW_PER_PASS) {
    OUTPUT(i_loc, tj) = SHARED(tj, i_loc);
  }
}
#undef T

// Launch the GPU kernel that would swap dimension-1 and dimension-2 in a
// 3D tensor. It looks at the shape of the incoming data, and decides the best
// strategy to launch.
template <typename T>
void RunSwapDimension1And2InTensor3(const T* input,
                                    const Dimension<3>& input_dims, T* output) {
  // If both dimensions are not trivial, use tiles for the actual swapping.
  // Otherwise, the trivial swapping relying on the ldg cache is more efficient.
  static const int kMinDimensionToUseTiles = 16;
  static const int TileSize = 32;
  static const int NumSubTiles = 8;
  bool use_tiles = (input_dims[1] >= kMinDimensionToUseTiles &&
                    input_dims[2] >= kMinDimensionToUseTiles);

  Dimension<3> input_dims_in_tiles = {
        input_dims[0], (input_dims[1] + TileSize - 1) / TileSize,
        (input_dims[2] + TileSize - 1) / TileSize,
  };

  int total_tiles_count = input_dims_in_tiles[0] * input_dims_in_tiles[1] *
                            input_dims_in_tiles[2];

  // int my_total_tiles_count = my_input_dims_in_tiles[0] * my_input_dims_in_tiles[1] *
  //                           my_input_dims_in_tiles[2];

  if (use_tiles) {
    // We get best performance when TileSize is the number of threads in a warp
    // (32 on our GPUs) and NumSubTiles is 8, so our block size is 8 * 32 = 256
    // threads.
    std::cout << "Swapping using tiles. "  << input_dims_in_tiles[0] << "," <<
                                                         input_dims_in_tiles[1] << "," <<
                                                         input_dims_in_tiles[2] << std::endl;
    dim3 griddim(total_tiles_count, 1, 1);
    dim3 blockdim(TileSize, NumSubTiles, 1);
    SwapDimension1And2InTensor3UsingTiles<T, TileSize, NumSubTiles><<<griddim, blockdim>>>(input, input_dims, output);
    // dim3 _blockdim(TILE_SIZE_I* TILE_SIZE_J, 1, 1);
    // MySwapDimension1And2InTensor3UsingTiles<<<
    //    griddim, THREAD_NUM>>>(input, input_dims, my_output);
  } else {
    std::cout << "Swapping using simple method." << std::endl;
    int total_element_count = input_dims[0] * input_dims[1] * input_dims[2];
    CudaLaunchConfig config = GetCudaLaunchConfig(total_element_count);
    SwapDimension1And2InTensor3Simple<T>
        <<<config.block_count, config.thread_per_block>>>(
            config.virtual_thread_count, input, input_dims, output);
    // MySwapDimension1And2InTensor3UsingTiles<<<
    //     my_total_tiles_count, THREAD_NUM>>>(input, input_dims, my_output);
  }
}

// Launch the GPU kernel that would swap dimension-1 and dimension-2 in a
// 3D tensor. It looks at the shape of the incoming data, and decides the best
// strategy to launch.
template <typename T>
void MyRunSwapDimension1And2InTensor3(const T* input,
                                    const Dimension<3>& input_dims, T* my_output) {
  // If both dimensions are not trivial, use tiles for the actual swapping.
  // Otherwise, the trivial swapping relying on the ldg cache is more efficient.
  static const int kMinDimensionToUseTiles = 16;
  static const int TileSize = 32;
  static const int NumSubTiles = 8;
  bool use_tiles = (input_dims[1] >= kMinDimensionToUseTiles &&
                    input_dims[2] >= kMinDimensionToUseTiles);

  Dimension<3> my_input_dims_in_tiles = {
        input_dims[0], (input_dims[1] + TILE_SIZE_I - 1) / TILE_SIZE_I,
        (input_dims[2] + TILE_SIZE_J - 1) / TILE_SIZE_J,
  };

  int my_total_tiles_count = my_input_dims_in_tiles[0] * my_input_dims_in_tiles[1] *
                            my_input_dims_in_tiles[2];

  MySwapDimension1And2InTensor3UsingTiles<<<
        my_total_tiles_count, THREAD_NUM>>>(input, input_dims, my_output);
}

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 } else { printf("Cuda launch success\n"); }\
}

int main()
{
#define N 128
#define M 512
#define P 32

  float *input_host, *output_host, *my_output_host;
  int size = N*M*P*sizeof(float);

  input_host = (float*)malloc(size);
  output_host = (float*)malloc(size);
  my_output_host = (float*)malloc(size);
  for (int i=0; i<N*M*P; i++) {
    input_host[i] = (float)i;
  }

  Dimension<3> input_dims = {
      N, M, P
  };

  float *input_device, *output_device, *my_output_device;
  cudaMalloc((void**)&input_device, size);
  cudaMalloc((void**)&output_device, size);
  cudaMalloc((void**)&my_output_device, size);
  cudaMemcpy(input_device, input_host, size, cudaMemcpyHostToDevice );
  RunSwapDimension1And2InTensor3(input_device, input_dims, output_device);
  MyRunSwapDimension1And2InTensor3(input_device, input_dims, my_output_device);
  cudaMemcpy(output_host, output_device, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(my_output_host, my_output_device, size, cudaMemcpyDeviceToHost);
  cudaCheckError();

  bool check = 1;
  for (int i=0; i<N; i++)
    for (int j=0; j<M; j++)
      for (int k=0; k<P; k++)
        check &= (output_host[i*M*P+j*P+k] == my_output_host[i*M*P+j*P+k]);

  printf("check %d\n", check);


  // for (int i=0; i<P; i++) {
  //   for (int j=0; j<M; j++) {
  //     std::cout.precision(5);
  //     std::cout << std::setw(6)   << output_host[i*M+j];
  //   }
  //   printf("\n");
  // }

  printf("===============\n");

  // for (int i=0; i<P; i++) {
  //   for (int j=0; j<M; j++) {
  //     std::cout.precision(5);
  //     std::cout << std::setw(6)   << my_output_host[i*M+j];
  //   }
  //   printf("\n");
  // }

  return 0;
}


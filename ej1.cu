#include "./common.h"
#include "./generator.cuh"

// calculate_sin: Calculates sin of x and y coordinates for points inside a square
__global__ void calculate_sin(unsigned int num_points, Point2D *points, double *sin_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point2D *point;
  point = &points[j + i * num_points];

  double x = point->x;
  double y = point->y;

  sin_result[j + i * num_points] = sin(x + y);
}

int main() {
  // Builds square of points with space of SMALL_POINT_SIZE
  unsigned int num_points_2d = trunc(SQUARE_LENGTH / SMALL_POINT_SIZE);

  size_t size_2d = num_points_2d * num_points_2d * sizeof(Point2D);

  Point2D *d_points_2d;
  CUDA_CHK(cudaMalloc((void**)&d_points_2d, size_2d));

  // dim3 block_dim(32, 32, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid_dim(num_points_2d / BLOCK_SIZE, num_points_2d / BLOCK_SIZE);

  // Generates points inside a square
  generate_square<<<grid_dim, block_dim>>>(d_points_2d, num_points_2d);
  CUDA_CHK(cudaGetLastError());

  CUDA_CHK(cudaDeviceSynchronize());

  Point2D *points_2d = (Point2D *)malloc(size_2d);
  CUDA_CHK(cudaMemcpy(points_2d, d_points_2d, size_2d, cudaMemcpyDeviceToHost));

  // Calculates sin of points
  double *d_sin_result;
  CUDA_CHK(cudaMalloc((void **)&d_sin_result, num_points_2d * num_points_2d * sizeof(double)));

  calculate_sin<<<grid_dim, block_dim>>>(num_points_2d, d_points_2d, d_sin_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *sin_result = (double *)malloc(num_points_2d * num_points_2d * sizeof(double));
  CUDA_CHK(cudaMemcpy(sin_result, d_sin_result, num_points_2d * num_points_2d * sizeof(double), cudaMemcpyDeviceToHost));

  printf("sin_result: with points %d \n", num_points_2d * num_points_2d);
  printf("Print only first X rows\n");

  // print_matrix_of_points_with_origin(points_2d, sin_result, 64);

  free(sin_result);
  free(points_2d);
  CUDA_CHK(cudaFree(d_sin_result));
  CUDA_CHK(cudaFree(d_points_2d));

  return 0;
}

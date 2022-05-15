#include "./common.h"
#include "./generator.cuh"

/*
  special_sum calculates the sum of all values inside a matrix in a radius.

  num_points: number of points in a matrix
  sum_result: pointer to the result array
  radius: radius of elements to sum
  matrix: pointer to the matrix with values
*/

// 196683650
// 39337986
__global__ void special_sum(unsigned int num_points, double *sum_result, int radius, double *matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double matrix_point[32][32];

  if (i >= num_points || j >= num_points) {
    return;
  }

  unsigned int threadIdx_x = threadIdx.x + 1;
  unsigned int threadIdx_y = threadIdx.y + 1;

  matrix_point[threadIdx_y -1][threadIdx_x -1] = matrix[i * num_points + j];
  __syncwarp();

  double result = -4 * matrix_point[threadIdx_y][threadIdx_x];
  for (int offset = -radius; offset <= radius; offset++) {
    for (int offset2 = -radius; offset2 <= radius; offset2++) {
      if (offset == offset2 || i + offset < 0 || i + offset >= num_points || j + offset2 < 0 || j + offset2 >= num_points) {
        continue;
      }

      if (-offset == offset2) {
        continue;
      }

      result += matrix_point[threadIdx_y + offset2][threadIdx_x + offset];
    }
  }

  sum_result[j + i * num_points] = result / SMALL_POINT_SIZE;
}

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
  unsigned int num_points_2d = trunc(SQUARE_LENGTH / SMALL_POINT_SIZE); // 6_280

  size_t size_2d = num_points_2d * num_points_2d * sizeof(Point2D); // 39_438_400 x double_size

  Point2D *d_points_2d;
  CUDA_CHK(cudaMalloc((void**)&d_points_2d, size_2d));

  // dim3 block_dim(32, 32, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid_dim(num_points_2d / BLOCK_SIZE, num_points_2d / BLOCK_SIZE);

  // Generates points inside a square
  generate_square<<<grid_dim, block_dim>>>(d_points_2d, num_points_2d);
  CUDA_CHK(cudaGetLastError());

  CUDA_CHK(cudaDeviceSynchronize());

  // Point3D *points_3d = (Point3D *)malloc(num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  // CUDA_CHK(cudaMemcpy(points_3d, d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyDeviceToHost));

  // Calculates sin of points
  double *d_sin_result;
  CUDA_CHK(cudaMalloc((void **)&d_sin_result, num_points_2d * num_points_2d * sizeof(double)));

  calculate_sin<<<grid_dim, block_dim>>>(num_points_2d, d_points_2d, d_sin_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *gpu_special_sum_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double)));

  special_sum<<<grid_dim, block_dim>>>(num_points_2d, gpu_special_sum_result, 1, d_sin_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *special_sum_result = (double *)malloc(num_points_2d * num_points_2d * sizeof(double));
  CUDA_CHK(cudaMemcpy(special_sum_result, gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double), cudaMemcpyDeviceToHost));

  // print_matrix_of_points(special_sum_result, 64);

  free(special_sum_result);
  CUDA_CHK(cudaFree(d_sin_result));
  CUDA_CHK(cudaFree(d_points_2d));
  CUDA_CHK(cudaFree(gpu_special_sum_result));

  return 0;
}

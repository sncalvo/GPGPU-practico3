#include "./common.h"
#include "./generator.cuh"

// calculate_sum_of_tan_xz: Calculates sum of tan for every column in a cube with number of points num_points
__global__ void calculate_sum_of_tan_xz(unsigned int num_points, Point3D *points, double *sum_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  Point3D *point;
  double sum = 0;
  for (unsigned int j = 0; j < num_points; j++) {
    point = &points[k * num_points * num_points + j * num_points + i];
    sum += tan(point->x + point->y + point->z);
  }

  sum_result[i + k * num_points] = sum;
}

int main() {
  // Builds cube of points with space of BIG_POINT_SIZE
  unsigned int num_points_3d = trunc(SQUARE_LENGTH / BIG_POINT_SIZE);

  Point3D *d_points_3d;
  CUDA_CHK(cudaMalloc(&d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D)));

  // Generates points inside a cube
  dim3 block_dim_cube(8, 8, 8);
  dim3 grid_dim_cube(num_points_3d / 8, num_points_3d / 8, num_points_3d / 8);
  generate_cube<<<grid_dim_cube, block_dim_cube>>>(d_points_3d, num_points_3d);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  // Point3D *points_3d = (Point3D *)malloc(num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  // CUDA_CHK(cudaMemcpy(points_3d, d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyDeviceToHost));

  // Execute tan calculation
  double *gpu_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_result, num_points_3d * num_points_3d * sizeof(double)));

  dim3 block_dim_matrix(32, 32, 1);
  dim3 grid_dim_matrix(num_points_3d / 32, num_points_3d / 32, 1);

  calculate_sum_of_tan_xz<<<grid_dim_matrix, block_dim_matrix>>>(num_points_3d, d_points_3d, gpu_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *result = (double *)malloc(num_points_3d * num_points_3d * sizeof(double));
  CUDA_CHK(cudaMemcpy(result, gpu_result, num_points_3d * num_points_3d * sizeof(double), cudaMemcpyDeviceToHost));

  // print_matrix_of_points(result, 8);

  free(result);
  CUDA_CHK(cudaFree(gpu_result));
  CUDA_CHK(cudaFree(d_points_3d));

  return 0;
}

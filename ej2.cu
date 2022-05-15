#include "./common.h"
#include "./generator.cuh"

// calculate_tan: Calculates tan for every point inside a cube with number of points num_points
__global__ void calculate_tan(unsigned int num_points, Point3D *points, double *tan_result) {
  unsigned int i = blockIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.z;

  __shared__ Point3D *point;
  point = &points[i + j * num_points + k * num_points * num_points];

  double x = point->x;
  double y = point->y;
  double z = point->z;

  tan_result[i + j * num_points + k * num_points * num_points] = tan(x + y + z);
}

int main() {
  // Builds cube of points with space of BIG_POINT_SIZE
  unsigned int num_points_3d = trunc(SQUARE_LENGTH / BIG_POINT_SIZE);

  Point3D *d_points_3d;
  CUDA_CHK(cudaMalloc(&d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D)));

  // Generates points inside a cube
  dim3 block_dim_cube(1, 1, 1);
  dim3 grid_dim_cube(num_points_3d, num_points_3d, num_points_3d);
  generate_cube<<<grid_dim_cube, block_dim_cube>>>(d_points_3d, num_points_3d);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  // Point3D *points_3d = (Point3D *)malloc(num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  // CUDA_CHK(cudaMemcpy(points_3d, d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyDeviceToHost));

  // Execute tan calculation
  double *gpu_tan_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_tan_result, num_points_3d * sizeof(double)));

  calculate_tan<<<grid_dim_cube, block_dim_cube>>>(num_points_3d, d_points_3d, gpu_tan_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *tan_result = (double *)malloc(num_points_3d * sizeof(double));
  CUDA_CHK(cudaMemcpy(tan_result, gpu_tan_result, num_points_3d * sizeof(double), cudaMemcpyDeviceToHost));

  print_matrix_of_points(tan_result, 8);

  free(tan_result);
  CUDA_CHK(cudaFree(gpu_tan_result));
  CUDA_CHK(cudaFree(d_points_3d));

  return 0;
}

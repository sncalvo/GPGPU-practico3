#include <math.h>
#include <iostream>

#include <stdlib.h>
#include "cuda.h"

#include <locale.h>

#define BLOCK_SIZE 16
#define PI 3.14159265358

#define SQUARE_LENGTH 2 * PI
#define SMALL_POINT_SIZE 0.001
#define BIG_POINT_SIZE 1.0
// #define BIG_POINT_SIZE 0.01

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Point2D {
  double x;
  double y;
};

struct Point3D {
  double x;
  double y;
  double z;
};

void print_matrix_of_points(Point2D *points, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << points[i * n + j].x << ", " << points[i * n + j].y << " ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
}

void print_matrix_of_points(Point3D *points, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        std::cout << points[i * n * n + j * n + k].x << ", " << points[i * n * n + j * n + k].y << ", " << points[i * n * n + j * n + k].z << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << std::endl;
}

void print_matrix_of_points(double *points, int n) {
  std::cout << "Begin printing matrix of points" << std::endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << points[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "End printing matrix of points" << std::endl;
}

void print_matrix_of_points3D(double *points, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        std::cout << points[i * n * n + j * n + k] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << std::endl;
}

__global__ void generate_square(Point2D *points, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    points[j + i * n] = {j * SMALL_POINT_SIZE - PI, i * SMALL_POINT_SIZE - PI};
  }
}

__global__ void generate_cube(Point3D *points, int n) {
  unsigned int i = blockIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.z;

  if (i < n && j < n && k < n) {
    points[i * n * n + j * n + k] = {i * BIG_POINT_SIZE - PI, j * BIG_POINT_SIZE - PI, k * BIG_POINT_SIZE - PI};
  }
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

// calculate_tan: Calculates tan for every point inside a cube with number of points num_points
__global__ void calculate_tan(unsigned int num_points, Point3D *points, double *tan_result) {
  unsigned int i = blockIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int k = blockIdx.z;

  __shared__ Point3D *point;
  point = &points[i + j * num_points + k * num_points * num_points];

  __syncthreads();

  double x = point->x;
  double y = point->y;
  double z = point->z;

  tan_result[i + j * num_points + k * num_points * num_points] = tan(x + y + z);
}

// calculate_sum_of_tan_xy: Calculates sum of tan for every row in a cube with number of points num_points
__global__ void calculate_sum_of_tan_xy(unsigned int num_points, Point3D *points, double *sum_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point3D *point;
  double sum = 0;
  for (unsigned int k = 0; k < num_points; k++) {
    point = &points[i + j * num_points + k * num_points * num_points];
    sum += tan(point->x + point->y + point->z);
  }

  sum_result[i + j * num_points] = sum;
}

// calculate_sum_of_tan_xz: Calculates sum of tan for every column in a cube with number of points num_points
__global__ void calculate_sum_of_tan_xz(unsigned int num_points, Point3D *points, double *sum_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point3D *point;
  double sum = 0;
  for (unsigned int j = 0; j < num_points; j++) {
    point = &points[i + j * num_points + k * num_points * num_points];
    sum += tan(point->x + point->y + point->z);
  }

  sum_result[i + k * num_points] = sum;
}

__global__ void calculate_sum_of_tan_yz(unsigned int num_points, Point3D *points, double *sum_result) {
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point3D *point;
  double sum = 0;
  for (unsigned int i = 0; i < num_points; i++) {
    point = &points[i + j * num_points + k * num_points * num_points];
    sum += tan(point->x + point->y + point->z);
  }

  sum_result[j + k * num_points] = sum;
}

/*
  special_sum calculates the sum of all values inside a matrix in a radius.

  num_points: number of points in a matrix
  sum_result: pointer to the result array
  radius: radius of elements to sum
  matrix: pointer to the matrix with values
*/
__global__ void special_sum(unsigned int num_points, double *sum_result, unsigned int radius, double *matrix) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ double *matrix_point;

  if (i >= num_points || j >= num_points) {
    return;
  }

  matrix_point = &matrix[j + i * num_points];

  double result = -4 * (*matrix_point);
  for (int offset = -radius; offset <= radius; offset++) {
    for (int offset2 = -radius; offset2 <= radius; offset2++) {
      if (offset == offset2 || i + offset < 0 || i + offset >= num_points || j + offset2 < 0 || j + offset2 >= num_points) {
        continue;
      }

      matrix_point = &matrix[(j + offset2) + (i + offset) * num_points];
      result += *matrix_point;
    }
  }

  sum_result[j + i * num_points] = result / SMALL_POINT_SIZE;
  // sum_result[j + i * num_points] = 11;
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

  // print_matrix_of_points(points_2d, 128);

  // Builds cube of points with space of BIG_POINT_SIZE
  unsigned int num_points_3d = trunc(SQUARE_LENGTH / BIG_POINT_SIZE);

  std::cout << "num_points_3d: " << num_points_3d << std::endl;

  Point3D *d_points_3d;
  CUDA_CHK(cudaMalloc(&d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D)));

  // Generates points inside a cube
  dim3 block_dim_cube(1, 1, 1);
  dim3 grid_dim_cube(num_points_3d, num_points_3d, num_points_3d);
  generate_cube<<<grid_dim_cube, block_dim_cube>>>(d_points_3d, num_points_3d);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  Point3D *points_3d = (Point3D *)malloc(num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  CUDA_CHK(cudaMemcpy(points_3d, d_points_3d, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyDeviceToHost));

  // print_matrix_of_points(points_3d, 8);

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

  print_matrix_of_points(sin_result, 64);

  // Calculates tan of points

  double *gpu_tan_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_tan_result, num_points_3d * sizeof(double)));

  calculate_tan<<<grid_dim_cube, block_dim_cube>>>(num_points_3d, d_points_3d, gpu_tan_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *tan_result = (double *)malloc(num_points_3d * sizeof(double));
  CUDA_CHK(cudaMemcpy(tan_result, gpu_tan_result, num_points_3d * sizeof(double), cudaMemcpyDeviceToHost));

  // print_matrix_of_points(tan_result, 8);

  // Calculates sum of tan for every row in a cube with number of points num_points
  double *gpu_sum_tan_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_sum_tan_result, num_points_3d * sizeof(double)));

  calculate_sum_of_tan_xz<<<grid_dim_cube, block_dim_cube>>>(num_points_3d, d_points_3d, gpu_sum_tan_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *sum_tan_result = (double *)malloc(num_points_3d * sizeof(double));
  CUDA_CHK(cudaMemcpy(sum_tan_result, gpu_sum_tan_result, num_points_3d * sizeof(double), cudaMemcpyDeviceToHost));

  // print_matrix_of_points(sum_tan_result, 8);

  /* Add other parts of ex 3 */
  double *gpu_special_sum_result;
  CUDA_CHK(cudaMalloc((void **)&gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double)));

  special_sum<<<grid_dim, block_dim>>>(num_points_2d, gpu_special_sum_result, 1, d_sin_result);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());

  double *special_sum_result = (double *)malloc(num_points_2d * num_points_2d * sizeof(double));
  CUDA_CHK(cudaMemcpy(special_sum_result, gpu_special_sum_result, num_points_2d * num_points_2d * sizeof(double), cudaMemcpyDeviceToHost));

  print_matrix_of_points(special_sum_result, 64);
  return 0;
}

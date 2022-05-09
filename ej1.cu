#define BLOCK_SIZE 32 // VEREMOS
#define SQUARE_POINTS (BLOCK_SIZE * BLOCK_SIZE)

struct Point2D {
  double x;
  double y;
};

struct Point3D {
  double x;
  double y;
  double z;
};

// calculate_sin: Calculates sin of x and y coordinates for points inside a square
__global__ void calculate_sin(unsigned int num_points, Point2D *points, double *sin_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point2D *point;
  point = points[i + j * num_points];

  __syncthreads();

  double x = point.x;
  double y = point.y;

  sin_result[i + j * num_points] = sin(x + y);
}

// calculate_tan: Calculates tan for every point inside a cube with number of points num_points
__global__ void calculate_tan(unsigned int num_points, Point3D *points, double *tan_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  __shared__ Point3D *point;
  point = points[i + j * num_points + k * num_points * num_points];

  __syncthreads();

  double x = point.x;
  double y = point.y;
  double z = point.z;

  tan_result[i + j * num_points + k * num_points * num_points] = tan(x + y + z);
}

// calculate_sum_of_tan_xy: Calculates sum of tan for every row in a cube with number of points num_points
__global__ void calculate_sum_of_tan_xy(unsigned int num_points, Point3D *points, double *sum_result) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point3D *point;
  double sum = 0;
  for (unsigned int k = 0; k < num_points; k++) {
    point = points[i + j * num_points + k * num_points * num_points];
    sum += tan(point.x + point.y + point.z);
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
    point = points[i + j * num_points + k * num_points * num_points];
    sum += tan(point.x + point.y + point.z);
  }

  sum_result[i + k * num_points] = sum;
}

__global__ void calculate_sum_of_tan_yz(unsigned int num_points, Point3D *points, double *sum_result) {
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point3D *point;
  double sum = 0;
  for (unsigned int i = 0; i < num_points; i++) {
    point = points[i + j * num_points + k * num_points * num_points];
    sum += tan(point.x + point.y + point.z);
  }

  sum_result[j + k * num_points] = sum;
}

__global__ void special_square_sum(unsigned int num_points, Point2D *points, double *sum_result, unsigned int radius) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ Point2D *point;

  point = points[i + j * num_points];
  int result = 4 * (point.x + point.y);
  for (int offset = -radius ; offset <= radius ; offset++) {
    for (int offset2 = -radius ; offset2 <= radius ; offset2++) {
      if (offset == 0 && offset2 == 0) {
        continue;
      }

      point = points[i + offset + (j + offset2) * num_points];
      result += point.x + point.y; // TODO: Change to function call
    }
  }

  sum_result[i + j * num_points] = result / SMALL_POINT_SIZE;
}

#define SQUARE_LENGTH 6.28318530718
#define SMALL_POINT_SIZE 0.001
#define BIG_POINT_SIZE 0.01

int main() {

  // Builds square of points with space of SMALL_POINT_SIZE
  double num_points_2d = SQUARE_LENGTH / SMALL_POINT_SIZE;
  Point2D *points = new Point2D[num_points_2d * num_points_2d];
  for (unsigned int i = 0; i < num_points_2d; i++) {
    for (unsigned int j = 0; j < num_points_2d; j++) {
      points[i + j * num_points_2d].x = i * SMALL_POINT_SIZE;
      points[i + j * num_points_2d].y = j * SMALL_POINT_SIZE;
    }
  }

  // Builds cube of points with space of BIG_POINT_SIZE
  double num_points_3d = SQUARE_LENGTH / BIG_POINT_SIZE;
  Point3D *points_cube = new Point3D[num_points_3d * num_points_3d * num_points_3d];
  for (unsigned int i = 0; i < num_points_3d; i++) {
    for (unsigned int j = 0; j < num_points_3d; j++) {
      for (unsigned int k = 0; k < num_points_3d; k++) {
        points_cube[i + j * num_points_3d + k * num_points_3d * num_points_3d].x = i * BIG_POINT_SIZE;
        points_cube[i + j * num_points_3d + k * num_points_3d * num_points_3d].y = j * BIG_POINT_SIZE;
        points_cube[i + j * num_points_3d + k * num_points_3d * num_points_3d].z = k * BIG_POINT_SIZE;
      }
    }
  }

  // Calculates sin of points
  cudaMalloc((void **)points, num_points_2d * num_points_2d * sizeof(Point2D));
  cudaMemcpy(points, points, num_points_2d * num_points_2d * sizeof(Point2D), cudaMemcpyHostToDevice);

  double *sin_result = new double[num_points_2d * num_points_2d];
  cudaMalloc((void **)&sin_result, num_points_2d * num_points_2d * sizeof(double));

  calculate_sin<<<num_points_2d, num_points_2d>>>(num_points_2d, points, sin_result);

  cudaMemcpy(sin_result, sin_result, num_points_2d * num_points_2d * sizeof(double), cudaMemcpyDeviceToHost);

  // Calculates tan of points
  cudaMalloc((void **)points, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  cudaMemcpy(points, points_cube, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyHostToDevice);

  double *tan_result = new double[num_points_3d * num_points_3d * num_points_3d];
  cudaMalloc((void **)&tan_result, num_points_3d * num_points_3d * num_points_3d * sizeof(double));

  calculate_tan<<<num_points_3d, num_points_3d, num_points_3d>>>(num_points_3d, points, tan_result);

  cudaMemcpy(tan_result, tan_result, num_points_3d * num_points_3d * num_points_3d * sizeof(double), cudaMemcpyDeviceToHost);

  // Calculates sum of sin for every row in a cube with number of points num_points
  cudaMalloc((void **)points, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  cudaMemcpy(points, points_cube, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyHostToDevice);

  double *sum_result = new double[num_points_3d * num_points_3d];
  cudaMalloc((void **)&sum_result, num_points_3d * num_points_3d * sizeof(double));

  calculate_sum_of_sin_xz<<<num_points_3d, num_points_3d>>>(num_points_3d, points, sum_result);

  cudaMemcpy(sum_result, sum_result, num_points_3d * num_points_3d * sizeof(double), cudaMemcpyDeviceToHost);

  // Calculates sum of tan for every row in a cube with number of points num_points
  cudaMalloc((void **)points, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  cudaMemcpy(points, points_cube, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyHostToDevice);

  double *sum_result_2 = new double[num_points_3d * num_points_3d];
  cudaMalloc((void **)&sum_result_2, *points, num_points_3d * num_points_3d * sizeof(double));

  calculate_sum_of_tan_xz<<<num_points_3d, num_points_3d>>>(num_points_3d, points, sum_result_2);

  cudaMemcpy(sum_result_2, sum_result_2, num_points_3d * num_points_3d * sizeof(double), cudaMemcpyDeviceToHost);

  // Calculates sum of sin for every row in a cube with number of points num_points
  cudaMalloc((void **)points, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D));
  cudaMemcpy(points, points_cube, num_points_3d * num_points_3d * num_points_3d * sizeof(Point3D), cudaMemcpyHostToDevice);

  double *sum_result_3 = new double[num_points_3d * num_points_3d];
  cudaMalloc((void **)&sum_result_3, num_points_3d * num_points_3d * sizeof(double));

  calculate_sum_of_sin_xy<<<num_points_3d, num_points_3d>>>(num_points_3d, points, sum_result_3);

  cudaMemcpy(sum_result_3, sum_result_3, num_points_3d * num_points_3d * sizeof(double), cudaMemcpyDeviceToHost);
}

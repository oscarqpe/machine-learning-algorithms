


#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

#define numCapas 3
#define  e  exp(1)

void forward (int *numNeuronasPorCapa, int * numFilasPorCapa, double *** MLP){
    
    for (int i = 1; i < numCapas; i++) {
        
        parallel_for (int j = 1; j < numNeuronasPorCapa[i]; j++) {
            
            MLP[i][j][numFilasPorCapa[i]-3] = 0 ;//Resetear NET
            
            parallel_for (int k = 0; k < numFilasPorCapa[i] -3 ; k++) {
                MLP[i][j][numFilasPorCapa[i]-3] += MLP[i-1][k][numFilasPorCapa[i-1]-2] * MLP[i][j][k];//NET
            }
            MLP[i][j][numFilasPorCapa[i]-2] = act_funct(MLP[i][j][numFilasPorCapa[i]-3]));
        }
    }
}

__global__ double act_funct(double val) {
    return 1.0/(1.0 + pow(e, - val);
}
                
                double dot_product (double *** MLP, int x1, int z1, int x2,int  y2, int dim) {
                    double sum = 0.0;
                    for (int k = 0; k < dim ; k++) {
                        sum =+  MLP [x1][k][z1] * MLP [x2][x2][k];
                        
                    }
                    return sum;
                }
                
__global__ void forward_cuda (int *numNeuronasPorCapa, int * numFilasPorCapa, double *** MLP){
    
    for (int i = 1; i < numCapas; i++) {
        for (int j = 1; j < numNeuronasPorCapa[i]; j++) {
            
            MLP[i][j][numFilasPorCapa[i]-3] = 0 ;//Resetear NET
            
         
            MLP[i][j][numFilasPorCapa[i]-3] = dot_product ( MLP, (i-1), (numFilasPorCapa[i-1]-2), i, j, numFilasPorCapa[i] -3);
            
            MLP[i][j][numFilasPorCapa[i]-2] = act_funct(MLP[i][j][numFilasPorCapa[i]-3]));//OUT
        }
    }
}


int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %fn", maxError);
}

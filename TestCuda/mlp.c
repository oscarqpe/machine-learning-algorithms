

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define  epsilon  exp(1)

#define NeuronByLayer 5
#define HiddenLayers 10
#define NeuronsByLayer (NeuronByLayer + 1)
#define outputSize  (NeuronByLayer + 4)

#define net(mlp, x,y,z)  mlp[ (x * HiddenLayers + y)*NeuronsByLayer + z ] 

 double act_funct(double val) {
    return 1.0/(1.0 + pow(epsilon, val));
}

void init (double * layer) {
     for (int h = 0; h < HiddenLayers; h++) {
        for (int i = 0; i < NeuronByLayer; i++) {
            for (int j = 0; j <= NeuronByLayer; j++) {
                float random = (10 + (rand() % 89));
                net(layer, h, i, j) = random / 100;
                net(layer, h, NeuronByLayer, j) = 0.0;
                layer[(h * 10 + NeuronByLayer) * NeuronsByLayer + j] = 0.0;
            }
        }
        net(layer,h,NeuronByLayer,NeuronByLayer + 2) = 1;
    }
} 

                
                                
void print_net (double * layer) {
     for (int h = 0; h < HiddenLayers; h++) {
        for (int i = 0; i < NeuronByLayer; i++) {
            for (int j = 0; j <= NeuronByLayer; j++) {
                 printf  ("%f, ", net(layer, h, i, j) ); 
            }
             printf  ("\n" ); 
        }
        printf  ("\n" ); 
    }
} 



void forward (int *numNeuronasPorCapa, int * numFilasPorCapa, double * MLP){
    
    for (int i = 1; i < HiddenLayers; i++) {
         for(int j=0;  j < numNeuronasPorCapa[i]; j++ ) {

            net(MLP, i, j, numFilasPorCapa[i]-3)  = 0.0;
            
             for (int k=0; k < numFilasPorCapa[i] -3; k++) {
                  net(MLP, i, j, numFilasPorCapa[i]-3) +=  net(MLP, i-1, k, numFilasPorCapa[i]-2) * net(MLP, i, j, k); 
            }
            net(MLP, i, j, numFilasPorCapa[i]-2)  =  act_funct(net(MLP, i, j, numFilasPorCapa[i]-3) ); 
        }
    }
}

int main () {
    double *MLP =  (double *)malloc (HiddenLayers * NeuronsByLayer * outputSize * sizeof(double));

    init(MLP);
    print_net(MLP);
   
    return 0;
    
}
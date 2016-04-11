#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

int main()
{
    srand(time(NULL));

    char buffer[1024] ;
    char *record,*line;
    int ii=0,jj=0;
    double data[610][59];
    FILE *fstream = fopen("\iris.data.csv","r");
    if(fstream == NULL)
    {
        printf("\n file opening failed ");
        return -1 ;
    }
    while (fgets(buffer, 1024, fstream))
    {
        char* tmp = strdup(buffer);
        const char* tok;
        for (tok = strtok(tmp, ";");
            tok && *tok;
			tok = strtok(NULL, ";\n"))
        {
            printf("Field  %s\n", tok);
        }

        // NOTE strtok clobbers tmp
        free(tmp);
    }

    int sizeOfInput = 4;
    int neuronByLayer = 4;
    int hiddenLayers = 2;
    int neuronByOutput = 1;
    double factorLearning = 0.5;
    // array to input
    double input[sizeOfInput];

    // add 3 rows to net, activate and delta errors
    double layer1[hiddenLayers][neuronByLayer][sizeOfInput + 3];

    // matrix to output
    double output[neuronByOutput][sizeOfInput + 3];

    // input data
    input[0] = 2;
    input[1] = 3;
    input[2] = 5;

    int h, i,j,k;

    /********************FORWARD PROPAGATION***********************/
    // forward to first layer
    for (i = 0; i < neuronByLayer; i++) {
        for (j = 0; j < sizeOfInput; j++) {
            //printf("%lf", layer1[i][j]);
            //printf(" ");
            if (layer1[0][i][j] <= 0.01) {
                //printf("i");
                layer1[0][i][j] = 0.01+(rand() % 1);
            }
            layer1[0][i][sizeOfInput] += input[j] * layer1[0][i][j];
        }
        layer1[0][i][sizeOfInput + 1] = (1 / ( 1 + exp((double) -1 * layer1[0][i][sizeOfInput])));
        layer1[0][i][sizeOfInput + 2] = 0.0;
    }

    // forward to hidden layers
    for (h = 1; h < hiddenLayers; h++) {
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j < sizeOfInput; j++) {
                //printf("%lf", layer1[i][j]);
                //printf(" ");
                if (layer1[h][i][j] <= 0.01) {
                    //printf("i");
                    layer1[h][i][j] = 0.01+(rand() % 1);
                }
                layer1[h][i][sizeOfInput] += layer1[h - 1][i][sizeOfInput + 1] * layer1[h][i][j];
            }
            layer1[h][i][sizeOfInput + 1] = (1 / ( 1 + exp((double) -1 * layer1[0][i][sizeOfInput])));
            layer1[h][i][sizeOfInput + 2] = 0.0;
        }
    }
    // fordward to outout layer
    for (i = 0; i < neuronByOutput; i++) {
        for (j = 0; j < sizeOfInput; j++) {
            if (output[i][j] <= 0.01) {
                //printf("i");
                output[i][j] = 0.01 + (rand() % 1);
            }
            output[i][sizeOfInput] += output[i][j] * layer1[hiddenLayers - 1][neuronByLayer - 1][j];
        }
        output[i][sizeOfInput + 1] = (1 / ( 1 + exp((double) -1 * output[i][sizeOfInput])));
        output[i][sizeOfInput + 2] = 0.0;
    }

    /*********************BACKWARD PROPAGATION********************/
    // calc delta error output layer
    for (i = 0; i < neuronByOutput; i++) {
        // error de la capa de salida que funcion utilizar ??????
        output[i][sizeOfInput + 2] = output[i][sizeOfInput + 1] - output[i][sizeOfInput];
    }
    // calc delta error from last hidden layers
    for (i = 0; i < neuronByLayer; i++) {
        for (j = 0; j < sizeOfInput; j++) {
            double sum = 0;
            for (k = 0; k < neuronByOutput; k++) {
                sum += output[k][sizeOfInput + 2] * output[k][j];
            }
            layer1[hiddenLayers - 1][i][sizeOfInput + 2] = sum;
        }
    }
    // calc delta error to hidden layers
    for (h = hiddenLayers - 2; h >= 0; h--) {
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j < sizeOfInput; j++) {
                double sum = 0;
                for (k = 0; k < neuronByLayer; k++) {
                    sum += layer1[h + 1][k][sizeOfInput + 2] * layer1[h + 1][k][j];
                }
                layer1[h][i][sizeOfInput + 2] = sum;
            }
        }
    }

    // UPDATE WEIGHTS
    // update weight first layer
    for (i = 0; i < sizeOfInput; i++) {
        for (j = 0; j < neuronByLayer; j++) {
            layer1[0][i][j] = layer1[0][i][j]
                            + factorLearning * layer1[0][i][sizeOfInput + 2]
                            * (layer1[0][i][sizeOfInput] * (1 - layer1[0][i][sizeOfInput]))
                            * input[i];
        }
    }

    // update weight hidden layers
    for (h = 1; h < hiddenLayers; h++) {
        for (i = 0; i < sizeOfInput; i++) {
            for (j = 0; j < neuronByLayer; j++) {
                layer1[h][i][j] = layer1[h][i][j]
                                + factorLearning * layer1[h][i][sizeOfInput + 2]
                                * (layer1[h][i][sizeOfInput] * (1 - layer1[h][i][sizeOfInput]))
                                * layer1[h - 1][i][sizeOfInput];
            }
        }
    }

    // update weight ouput layer
    for (i = 0; i < neuronByLayer; i++) {
        for (j = 0; j < neuronByOutput; j++) {
            output[i][j] = output[i][j] + factorLearning * output[i][sizeOfInput + 2]
                            * (output[i][sizeOfInput] * (1 - output[i][sizeOfInput])) * layer1[hiddenLayers - 1][i][sizeOfInput];
        }
    }

    /**********************PRINT NETWORK NEURON**********************************/
    printf("\n");

    // print layers
    for (h = 0; h < hiddenLayers; h++) {
        printf("Layer");
        printf("%d", h);
        printf("\n");
        for (j = 0; j < sizeOfInput + 3; j++) {
            for (i = 0; i < neuronByLayer; i++) {
                printf("%lf", layer1[h][i][j]);
                printf("\t");
            }
            printf("\n");
        }
        printf("\n");
    }
    // print output
    printf("OUTPUT\n");
    for (j = 0; j < sizeOfInput + 3; j++) {
        for (i = 0; i < neuronByOutput; i++) {
            printf("%lf", output[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    return 0;
}

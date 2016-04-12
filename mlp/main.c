#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#ifdef NAN
#endif
int main()
{
    time_t t;
    /* Intializes random number generator */
    srand((unsigned) time(&t));
    int totalRows = 151;

    int sizeOfInput = 4;
    int neuronByLayer = 5;
    int hiddenLayers = 2;
    int neuronByOutput = 3;
    double factorLearning = 0.5;


    char buffer[4096] ;
    char *record,*line;
    int ii=0,jj=0;
    double data[totalRows][sizeOfInput + 1];
    FILE *fstream = fopen("\iris.norm.csv","r");
    if(fstream == NULL)
    {
        printf("\n file opening failed ");
        return -1 ;
    }
    while (fgets(buffer, 4096, fstream))
    {
        char* tmp = strdup(buffer);
        const char* tok;
<<<<<<< HEAD
        for (tok = strtok(tmp, ";"); tok && *tok; tok = strtok(NULL, ";\n"))
=======
        jj = 0;
        for (tok = strtok(tmp, ";");
            tok && *tok;
			tok = strtok(NULL, ";\n"))
>>>>>>> 0a15d005d034d4826073d8295611c2aa35da9c7c
        {
            //printf("Field  %s\n", tok);
            data[ii][jj] = atof(tok);
            //printf("I  %d\n", ii);
            //printf("J  %d\n", jj);
            jj++;
        }
        ii++;
        // NOTE strtok clobbers tmp
        free(tmp);
    }


    // array to input
    double input[sizeOfInput];

    // add 3 rows to net, activate and delta errors
    double layer1[hiddenLayers][neuronByLayer][neuronByLayer + 3];

    // matrix to output
    double output[neuronByOutput][sizeOfInput + 3];

    // input data
    //input[0] = 0.2;
    //input[1] = 0.3;
    //input[2] = 0.5;
    //input[3] = 0.5;

    int h, i,j,k, l;
    int epoch;
    for (epoch = 100000; epoch > 0; epoch--)
    for (l = 0; l < totalRows; l++) {
        //printf("Training ");
        //printf("%d", l + 1);
        // copy each row of data to input
        for (i = 0; i < sizeOfInput; i++) {
            input[i] = data[l][i];
            //printf(": %fl\t", input[i]);
        }
        //printf("\n");
        /********************FORWARD PROPAGATION***********************/
        // forward to first layer
        for (i = 0; i < neuronByLayer; i++) {
            layer1[0][i][neuronByLayer] = 0.0;
            for (j = 0; j < sizeOfInput; j++) {
                //printf("%lf", layer1[i][j]);
                //printf(" ");
                if (layer1[0][i][j] <= 0.01 || isnan(layer1[0][i][j])) {
                    float xd = (10 + (rand() % 89));
                    layer1[0][i][j] = xd / 100;
                }
                layer1[0][i][neuronByLayer] += input[j] * layer1[0][i][j];
            }
            layer1[0][i][neuronByLayer + 1] = (1 / ( 1 + exp((double) -1 * layer1[0][i][neuronByLayer])));
            layer1[0][i][neuronByLayer + 2] = 0.0;
        }

        // forward to hidden layers
        for (h = 1; h < hiddenLayers; h++) {
            for (i = 0; i < neuronByLayer; i++) {
                layer1[h][i][neuronByLayer] = 0.0;
                for (j = 0; j < neuronByLayer; j++) {
                    //printf("%lf", layer1[i][j]);
                    //printf(" ");
                    if (layer1[h][i][j] <= 0.01 || isnan(layer1[h][i][j])) {
                        float xd = (10 + (rand() % 89));
                        layer1[h][i][j] = xd / 100;
                    }
                    layer1[h][i][neuronByLayer] += layer1[h - 1][i][neuronByLayer + 1] * layer1[h][i][j];
                }
                layer1[h][i][neuronByLayer + 1] = (1 / ( 1 + exp((double) -1 * layer1[0][i][neuronByLayer])));
                layer1[h][i][neuronByLayer + 2] = 0.0;
            }
        }

        // fordward to outout layer
        for (i = 0; i < neuronByOutput; i++) {
            output[i][neuronByLayer] = 0;
            for (j = 0; j < neuronByLayer; j++) {
                if (output[i][j] <= 0.01 || isnan(output[i][j])) {
                    float xd = (10 + (rand() % 89));
                    output[i][j] = xd / 100;
                }
                output[i][neuronByLayer] += output[i][j] * layer1[hiddenLayers - 1][neuronByLayer - 1][j];
            }
            output[i][neuronByLayer + 1] = (1 / ( 1 + exp((double) -1 * output[i][neuronByLayer])));
            output[i][neuronByLayer + 2] = 0.0;
        }

        /*********************BACKWARD PROPAGATION********************/

        // calc delta error output layer
        for (i = 0; i < neuronByOutput; i++) {
            // error de la capa de salida que funcion utilizar ??????
            if (data[l][4] == 0) {
                if (i == 0) {
                    output[i][neuronByLayer + 2] = 1 - output[i][neuronByLayer + 1];
                }
                else {
                    output[i][neuronByLayer + 2] = 0 - output[i][neuronByLayer + 1];
                }
            }
            else if (data[l][4] == 1) {
                if (i == 1) {
                    output[i][neuronByLayer + 2] = 1 - output[i][neuronByLayer + 1];
                }
                else {
                    output[i][neuronByLayer + 2] = 0 - output[i][neuronByLayer + 1];
                }
            }
            else if (data[l][4] == 2) {
                if (i == 2) {
                    output[i][neuronByLayer + 2] = 1 - output[i][neuronByLayer + 1];
                }
                else {
                    output[i][neuronByLayer + 2] = 0 - output[i][neuronByLayer + 1];
                }
            }
        }

        // calc delta error from last hidden layers
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j < neuronByLayer; j++) {
                double sum = 0;
                for (k = 0; k < neuronByOutput; k++) {
                    sum += output[k][neuronByLayer + 2] * output[k][j];
                }
                layer1[hiddenLayers - 1][i][neuronByLayer + 2] = sum;
            }
        }

        // calc delta error to hidden layers
        for (h = hiddenLayers - 2; h >= 0; h--) {
            for (i = 0; i < neuronByLayer; i++) {
                for (j = 0; j < neuronByLayer; j++) {
                    double sum = 0;
                    for (k = 0; k < neuronByLayer; k++) {
                        sum += layer1[h + 1][k][neuronByLayer + 2] * layer1[h + 1][k][j];
                    }
                    layer1[h][i][neuronByLayer + 2] = sum;
                }
            }
        }

        // UPDATE WEIGHTS
        // update weight first layer
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j < sizeOfInput; j++) {
                layer1[0][i][j] +=
                                factorLearning * layer1[0][i][neuronByLayer + 2]
                                * (layer1[0][i][neuronByLayer + 1] * (1 - layer1[0][i][neuronByLayer + 1]))
                                * input[i];
            }
        }

        // update weight hidden layers
        for (h = 1; h < hiddenLayers; h++) {
            for (i = 0; i < neuronByLayer; i++) {
                for (j = 0; j < neuronByLayer; j++) {
                    layer1[h][i][j] +=
                                    factorLearning * layer1[h][i][neuronByLayer + 2]
                                    * (layer1[h][i][neuronByLayer + 1] * (1 - layer1[h][i][neuronByLayer + 1]))
                                    * layer1[h - 1][i][neuronByLayer + 1];
                }
            }
        }

        // update weight ouput layer
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j < neuronByOutput; j++) {
                output[j][i] +=
                                factorLearning * output[j][neuronByLayer + 2]
                                * (output[j][neuronByLayer + 1] * (1 - output[j][neuronByLayer + 1]))
                                * layer1[hiddenLayers - 1][i][neuronByLayer + 1];
            }
        }

    }
    /**********************PRINT NETWORK NEURON**********************************/
    printf("\n");

    // print layers
    for (h = 0; h < hiddenLayers; h++) {
        printf("Layer");
        printf("%d", h);
        printf("\n");
        for (j = 0; j < neuronByLayer + 3; j++) {
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
    for (j = 0; j < neuronByLayer + 3; j++) {
        for (i = 0; i < neuronByOutput; i++) {
            printf("%lf", output[i][j]);
            printf("\t");
        }
        printf("\n");
    }
    return 0;
}

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
    int totalRows = 135;

    int sizeOfInput = 4;
    int neuronByLayer = 4;
    int hiddenLayers = 3;
    int neuronByOutput = 3;
    double factorLearning = 0.05;


    char buffer[4096] ;
    char *record,*line;
    int ii=0,jj=0;
    double data[totalRows][sizeOfInput + 3];
    double test[15][sizeOfInput + 3];
    FILE *fstream = fopen("\iris.training.csv","r");
    if(fstream == NULL)
    {
        printf("\n file training opening failed ");
        return -1 ;
    }
    FILE *fstreamTest = fopen("\iris.test.csv","r");
    if(fstreamTest == NULL)
    {
        printf("\n file test opening failed ");
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
            jj++;
        }
        // NOTE strtok clobbers tmp
        free(tmp);
        ii++;
    }
    ii = 0; jj = 0;
    while (fgets(buffer, 4096, fstreamTest))
    {
        char* tmp = strdup(buffer);
        const char* tok;
        jj = 0;
        for (tok = strtok(tmp, ";");
            tok && *tok;
			tok = strtok(NULL, ";\n"))
        {
            //printf("Field  %s\n", tok);
            test[ii][jj] = atof(tok);
            jj++;
        }
        // NOTE strtok clobbers tmp
        free(tmp);
        ii++;
    }


    // array to input
    double input[sizeOfInput + 1];

    // add 3 rows to net, activate and delta errors
    double layer1[hiddenLayers][neuronByLayer + 1][neuronByLayer + 4];

    // matrix to output
    double output[neuronByOutput][neuronByLayer + 4];

    // input data
    //input[0] = 0.2;
    //input[1] = 0.3;
    //input[2] = 0.5;
    //input[3] = 0.5;

    int h, i,j,k, l;
    int epoch;

    // initialze weight in hidden layers
    for (h = 0; h < hiddenLayers; h++) {
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j <= neuronByLayer; j++) {
                float random = (10 + (rand() % 89));
                layer1[h][i][j] = random / 100;
                layer1[h][neuronByLayer][j] = 0.0;
            }
        }
        layer1[h][neuronByLayer][neuronByLayer + 2] = 1;
    }
    // initialize weights to outout layer
    for (i = 0; i < neuronByOutput; i++) {
        for (j = 0; j <= neuronByLayer; j++) {
            float random = (10 + (rand() % 89));
            output[i][j] = random / 100;
        }
    }

    for (epoch = 1; epoch > 0; epoch--)
    for (l = 0; l < totalRows; l++) {
        //printf("Training ");
        //printf("%d", l + 1);
        // copy each row of data to input
        for (i = 0; i < sizeOfInput; i++) {
            input[i] = data[l][i];
            //printf(": %fl\t", input[i]);
        }
        input[sizeOfInput] = 1;
        //printf("\n");
        /********************FORWARD PROPAGATION***********************/
        // forward to first layer

        for (i = 0; i < neuronByLayer; i++) {
            layer1[0][i][neuronByLayer + 1] = 0.0;
            for (j = 0; j <= sizeOfInput; j++) {
                layer1[0][i][neuronByLayer + 1] += input[j] * layer1[0][i][j];
            }
            layer1[0][i][neuronByLayer + 2] = (1 / ( 1 + exp((double) -1 * layer1[0][i][neuronByLayer + 1])));
            layer1[0][i][neuronByLayer + 3] = 0.0;
        }

        // forward to hidden layers
        for (h = 1; h < hiddenLayers; h++) {
            for (i = 0; i < neuronByLayer; i++) {
                layer1[h][i][neuronByLayer + 1] = 0.0;
                for (j = 0; j <= neuronByLayer; j++) {
                    layer1[h][i][neuronByLayer + 1] += layer1[h - 1][i][neuronByLayer + 2] * layer1[h][i][j];
                }
                layer1[h][i][neuronByLayer + 2] = (1 / ( 1 + exp((double) -1 * layer1[h][i][neuronByLayer + 1])));
                layer1[h][i][neuronByLayer + 3] = 0.0;
            }
        }

        // fordward to outout layer
        for (i = 0; i < neuronByOutput; i++) {
            output[i][neuronByLayer + 1] = 0.0;
            for (j = 0; j <= neuronByLayer; j++) {
                output[i][neuronByLayer + 1] += output[i][j] * layer1[hiddenLayers - 1][j][neuronByLayer + 2];
            }
            output[i][neuronByLayer + 2] = (1 / ( 1 + exp((double) -1 * output[i][neuronByLayer + 1])));
            output[i][neuronByLayer + 3] = 0.0;
        }

        /*********************BACKWARD PROPAGATION********************/

        // calc delta error output layer
        for (i = 0; i < neuronByOutput; i++) {
            // error de la capa de salida que funcion utilizar ??????
            printf("%fl\n", data[l][sizeOfInput + i]);
            output[i][neuronByLayer + 3] = data[l][sizeOfInput + i] - output[i][neuronByLayer + 2];
        }

        // calc delta error from last hidden layers
        //for (i = 0; i <= neuronByLayer; i++) {// iterator to delta in neuron on the last hidden layer
            for (j = 0; j <= neuronByLayer; j++) { // iterator to weights of the neuron output
                double sum = 0;
                for (k = 0; k < neuronByOutput; k++) { // iterator to neuron by neuron in output layer
                    sum += output[k][neuronByLayer + 3] * output[k][j];
                }
                layer1[hiddenLayers - 1][j][neuronByLayer + 3] = sum;
            }
        //}

        // calc delta error to hidden layers
        for (h = hiddenLayers - 2; h >= 0; h--) { // iterator from second last to first layer
            //for (i = 0; i < neuronByLayer; i++) { //
                for (j = 0; j <= neuronByLayer; j++) {
                    double sum = 0;
                    for (k = 0; k <= neuronByLayer; k++) {
                        sum += layer1[h + 1][k][neuronByLayer + 3] * layer1[h + 1][k][j];
                    }
                    layer1[h][j][neuronByLayer + 3] = sum;
                }
            //}
        }

        // UPDATE WEIGHTS
        // update weight first layer
        for (i = 0; i < neuronByLayer; i++) {
            for (j = 0; j <= sizeOfInput; j++) {
                layer1[0][i][j] +=
                                factorLearning * layer1[0][i][neuronByLayer + 3]
                                * (layer1[0][i][neuronByLayer + 2] * (1 - layer1[0][i][neuronByLayer + 2]))
                                * input[j];
            }
        }

        // update weight hidden layers
        for (h = 1; h < hiddenLayers; h++) {
            for (i = 0; i < neuronByLayer; i++) {
                for (j = 0; j <= neuronByLayer; j++) {
                    layer1[h][i][j] +=
                                    factorLearning * layer1[h][i][neuronByLayer + 3]
                                    * (layer1[h][i][neuronByLayer + 2] * (1 - layer1[h][i][neuronByLayer + 2]))
                                    * layer1[h - 1][j][neuronByLayer + 2];
                }
            }
        }

        // update weight ouput layer
        for (i = 0; i <= neuronByLayer; i++) {
            for (j = 0; j < neuronByOutput; j++) {
                output[j][i] +=
                                factorLearning * output[j][neuronByLayer + 3]
                                * (output[j][neuronByLayer + 2] * (1 - output[j][neuronByLayer + 2]))
                                * layer1[hiddenLayers - 1][i][neuronByLayer + 2];
            }
        }
        //printf("%fl\n", output[0][0]);
    }
    /**********************PRINT NETWORK NEURON**********************************/
    printf("\n");

    // print layers
    for (h = 0; h < hiddenLayers; h++) {
        printf("Layer");
        printf("%d", h);
        printf("\n");
        for (j = 0; j < neuronByLayer + 4; j++) {
            for (i = 0; i <= neuronByLayer; i++) {
                printf("%lf", layer1[h][i][j]);
                printf("\t");
            }
            printf("\n");
        }
        printf("\n");
    }
    // print output
    printf("OUTPUT\n");
    for (j = 0; j < neuronByLayer + 4; j++) {
        for (i = 0; i < neuronByOutput; i++) {
            printf("%lf", output[i][j]);
            printf("\t");
        }
        printf("\n");
    }

    // TESTING
    printf("\nTESTING\n");

    for (l = 0; l < 15; l++) {
        //printf("Training ");
        //printf("%d", l + 1);
        // copy each row of data to input
        printf("\nEnter:\n");
        for (i = 0; i < sizeOfInput; i++) {
            input[i] = test[l][i];
            printf("%fl\t", input[i]);
        }
        printf("\n");
        input[sizeOfInput] = 1;
        //printf("\n");
        /********************FORWARD PROPAGATION***********************/
        // forward to first layer

        for (i = 0; i < neuronByLayer; i++) {
            layer1[0][i][neuronByLayer + 1] = 0.0;
            for (j = 0; j <= sizeOfInput; j++) {
                layer1[0][i][neuronByLayer + 1] += input[j] * layer1[0][i][j];
            }
            layer1[0][i][neuronByLayer + 2] = (1 / ( 1 + exp((double) -1 * layer1[0][i][neuronByLayer + 1])));
            //layer1[0][i][neuronByLayer + 3] = 0.0;
        }

        // forward to hidden layers
        for (h = 1; h < hiddenLayers; h++) {
            for (i = 0; i < neuronByLayer; i++) {
                layer1[h][i][neuronByLayer + 1] = 0.0;
                for (j = 0; j <= neuronByLayer; j++) {
                    layer1[h][i][neuronByLayer + 1] += layer1[h - 1][i][neuronByLayer + 2] * layer1[h][i][j];
                }
                layer1[h][i][neuronByLayer + 2] = (1 / ( 1 + exp((double) -1 * layer1[h][i][neuronByLayer + 1])));
                //layer1[h][i][neuronByLayer + 3] = 0.0;
            }
        }

        // fordward to outout layer
        for (i = 0; i < neuronByOutput; i++) {
            output[i][neuronByLayer + 1] = 0.0;
            for (j = 0; j <= neuronByLayer; j++) {
                output[i][neuronByLayer + 1] += output[i][j] * layer1[hiddenLayers - 1][j][neuronByLayer + 2];
            }
            output[i][neuronByLayer + 2] = (1 / ( 1 + exp((double) -1 * output[i][neuronByLayer + 1])));
            //output[i][neuronByLayer + 3] = 0.0;
        }

        // calc delta error output layer
        for (i = 0; i < neuronByOutput; i++) {
            // error de la capa de salida que funcion utilizar ??????
            printf("%fl\t", output[i][neuronByLayer + 2]);
            //output[i][neuronByLayer + 3] = data[l][sizeOfInput + i] - output[i][neuronByLayer + 2];
        }
        printf("\n");
        // print layers
        /*for (h = 0; h < hiddenLayers; h++) {
            printf("Layer");
            printf("%d", h);
            printf("\n");
            for (j = 0; j < neuronByLayer + 4; j++) {
                for (i = 0; i <= neuronByLayer; i++) {
                    printf("%lf", layer1[h][i][j]);
                    printf("\t");
                }
                printf("\n");
            }
            printf("\n");
        }
        // print output
        printf("OUTPUT\n");
        for (j = 0; j < neuronByLayer + 4; j++) {
            for (i = 0; i < neuronByOutput; i++) {
                printf("%lf", output[i][j]);
                printf("\t");
            }
            printf("\n");
        }*/
    }
    return 0;
}

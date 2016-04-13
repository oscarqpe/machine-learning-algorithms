//
//  main.cpp
//  PunterosVoid RedNeuronal
//
//  Created by Andre Valdivia on 12/04/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <stdlib.h>



#define epsilon exp(1)

#define NeuronByLayer 5

#define HiddenLayers 3
#define NeuronsByLayer (NeuronByLayer + 1)
#define outputSize  (NeuronByLayer + 4)

//#define net(mlp, x,y,z)  mlp[(x * NeuronsByLayer + y) * outputSize + z]
#define net(mlp, x,y,z)  mlp[(z * NeuronsByLayer + y) * HiddenLayers + x]
using namespace std;
int main(int argc, const char * argv[]) {

/////////////////////////////////////////////////
///////////   INICIALIZACION DE MLP   ///////////
/////////////////////////////////////////////////

    double e = exp(1);

    int numDatosEntrada = 2;
    int numDatosSalida = 2;
    int numCapas = 3;

    double real[numDatosSalida + 1];
    double error[numDatosSalida + 1];

    double lRate = 0.5;
    //cout<< "Ingrese el learning rate: ";
    //cin>> lRate;

    //cout<<"Ingrese el numero de capas intermedias: ";
    //cin >> numCapas;
    //numCapas += 2;
    int numNeuronasPorCapa[numCapas];
    //    int numNeuronasPorCapaAnterior[numCapas + 1];
    int numFilasPorCapa[numCapas];
    //double*** MLP;
    //MLP = (double***) malloc(sizeof(double)* numCapas);
    //Inicializar numCapas

    //Capa de entrada. Numero de datos de entrada mas el bias
    //MLP[0] = (double**)malloc(sizeof(double)*numDatosEntrada + 1);
    double *MLP =  (double *)malloc(HiddenLayers * NeuronsByLayer * outputSize * sizeof(double));


    numNeuronasPorCapa[0] = numDatosEntrada + 1;
    for (int i = 0; i < numNeuronasPorCapa[0]; i++) {
        // 3 Filas por neurona
        //MLP[0][i] = (double*)malloc(sizeof(double)*3);
        for (int j = 0; j < 3; j++) {
            //MLP[0][i][j] = 0;
            net(MLP, 0, i, j)  = 0.0;
        }
    }

    //Capas intermedias
    for (int i = 1; i<numCapas-1; i++) {
        int numNeuronas;// = NeuronByLayer;
        cout<<"Ingrese el numero de neuronas en la capa " << i <<": ";
        cin >> numNeuronas;
        numNeuronas += 1;
        numNeuronasPorCapa[i] = numNeuronas;
        //Neuronas por capa, ya se le sumo una del bias
        //MLP[i] = (double**)malloc(sizeof(double)*numNeuronas);
        for (int j = 0; j < numNeuronas; j++) {
            //Filas por neurona por capa mas 3 del net, out y delta
            //MLP[i][j] = (double *)malloc(sizeof(double)*(numNeuronasPorCapa[i-1] + 3));
            for (int k = 0; k < numNeuronasPorCapa[i-1] + 3; k++) {
//                cout<<"Num neuronas anterior: "<<numNeuronasPorCapa[i]<<" "<< i << "-"<<j<<"-"<<k<<endl;
                //MLP[i][j][k] = 0;
                net(MLP, i, j, k) = 0;
            }
        }
    }

    //Capa de salida
    numNeuronasPorCapa[numCapas - 1] = numDatosSalida + 1;
    //MLP[numCapas-1] = (double **)malloc(sizeof(double)*numNeuronasPorCapa[numCapas-1]);
    for (int i = 0; i < numNeuronasPorCapa[numCapas - 1]; i++) {
        //MLP[numCapas-1][i] = (double *)malloc(sizeof(double) * (numNeuronasPorCapa[numCapas-2]+3));
        for (int j = 0; j < numNeuronasPorCapa[numCapas - 2] + 3; j++) {
            //MLP[numCapas-1][i][j] = 0;
            net(MLP, numCapas - 1, i, j) = 0.0;
//            cout<<"Num neuronas anterior: "<<numNeuronasPorCapa[numCapas-1]<<" "<< numCapas-1 << "-"<<i<<"-"<<j<<endl;
        }
    }

    //TRUQUITOS JIJI
//    numNeuronasPorCapaAnterior[0] = 0;
//    for (int i = 1 ; i < numCapas + 1; i++) {
//        numNeuronasPorCapaAnterior[i] = numNeuronasPorCapa[i-1];
//    }

    numFilasPorCapa[0] = 3;
    for (int i = 1 ; i < numCapas + 1; i++) {
        numFilasPorCapa[i] = numNeuronasPorCapa[i-1] + 3;
        cout<< numFilasPorCapa[i]  << endl;
    }

/////////////////////////////////////////////////
/////////////  SETEAR MATRIZ    /////////////////
/////////////////////////////////////////////////

//    Inicializar bias
    for (int i = 0 ; i < numCapas; i++) {
//        MLP[i][0][numNeuronasPorCapaAnterior[i] + 1] = 1;
        //MLP[i][0][numFilasPorCapa[i] - 2] = 1;
        cout << i << ", " << 0 << ", " << numFilasPorCapa[i] - 2 << endl;
        net(MLP, i, 0, numFilasPorCapa[i] - 2) = 1;
    }
/*
    //Setear la matriz de ejemplo
    net(MLP, 0, 1, 1) = 0.05;
    net(MLP, 0, 2, 1) = 0.10;

    net(MLP, 1, 1, 0) = 0.35;//b1
    net(MLP, 1, 2, 0) = 0.35;//b1
    net(MLP, 1, 1, 1) = 0.15;//w1
    net(MLP, 1, 1, 2) = 0.20;//w2
    net(MLP, 1, 2, 1) = 0.25;//w3
    net(MLP, 1, 2, 2) = 0.30;//w4

    net(MLP, 2, 1, 0) = 0.60;//b2
    net(MLP, 2, 2, 0) = 0.60;//b2
    net(MLP, 2, 1, 1) = 0.40;//w5
    net(MLP, 2, 1, 2) = 0.45;//w6
    net(MLP, 2, 2, 1) = 0.50;//w7
    net(MLP, 2, 2, 2) = 0.55;//w8

    //Setear matrices aleatoreas
//    for (int i = 0; i < numCapas; i++) {
//        for (int j = 1; j < numNeuronasPorCapa[i]; j++) {
//            for (int k = 0; k < numNeuronasPorCapaAnterior[i]; k++) {
//                MLP[i][j][k] = 3333;//Poner random aqui
//            }
//        }
//    }

    real[1] = 0.01;
    real[2] = 0.99;

/////////////////////////////////////////////////
////////////////  FORWARD    ////////////////////
/////////////////////////////////////////////////
/*
    for (int i = 1; i < numCapas; i++) {
        for (int j = 1; j < numNeuronasPorCapa[i]; j++) {

            net(MLP, i, j, numFilasPorCapa[i]-3) = 0 ;//Resetear NET

            for (int k = 0; k < numFilasPorCapa[i] -3 ; k++) {
                net(MLP, i, j, numFilasPorCapa[i]-3) +=
                    net(MLP, i-1, k, numFilasPorCapa[i-1]-2) *
                    net(MLP, i, j, k); //NET
            }
            net(MLP, i, j, numFilasPorCapa[i]-2) = 1/(1 + pow(e, - net(MLP, i, j, numFilasPorCapa[i]-3)));//OUT
        }
    }
/*
/////////////////////////////////////////////////
////////////////  ERROR    //////////////////////
/////////////////////////////////////////////////

    error[0] = 0;
    for (int i = 1; i <= numDatosSalida; i++) {
        error[i] = pow(real[i] -
                   net(MLP, numCapas-1, i, numFilasPorCapa[numCapas-1]-2), 2)/2 ;
        error[0] += error[i];
    }

/////////////////////////////////////////////////
////////////////  BACKWARD    ///////////////////
/////////////////////////////////////////////////

/*
    //Delta de la ultima capa

    for (int i = 1; i < numNeuronasPorCapa[numCapas-1]; i++) {
        double a = -(real[i] - MLP[numCapas-1][i][numFilasPorCapa[numCapas-1]-2]);
        double b = MLP[numCapas-1][i][numFilasPorCapa[numCapas-1]-2] * ( 1 - MLP[numCapas-1][i][numFilasPorCapa[numCapas-1]-2]);
        MLP[numCapas-1][i][numFilasPorCapa[numCapas-1]-1] =  a * b;
    }

    //Delta de las capas intermedias
    for (int i = numCapas-2; i > 0; i--) {
        //Recorre columnas
        for (int j = 1 ; j < numNeuronasPorCapa[i];j++  ) {
            double a = net(MLP, i, j, numFilasPorCapa[i]-2) *
                (1 - net(MLP, i, j, numFilasPorCapa[i]-2));
            double b = 0;
            for (int k = 1; k < numNeuronasPorCapa[i+1]; k++) {
                b += net(MLP, i+1, k, numFilasPorCapa[i+1]-1) *
                net(MLP, i + 1, k, j);
            }
            net(MLP, i, j, numFilasPorCapa[i]-1) = a * b;
        }
    }

    //Actualizar los pesos
    for (int i = 1; i < numCapas; i++) {
        for (int j = 1; j < numNeuronasPorCapa[i]; j++) {
            for (int k = 0; k < numFilasPorCapa[i]-3; k++) {
                double anterior = net(MLP, i, j, k);
                double delta = net(MLP, i, j, numFilasPorCapa[i]-1);
                double out = net(MLP, i-1, k, numFilasPorCapa[i-1]-2);

                net(MLP, i, j, k) = anterior - lRate * ( delta * out);
            }
        }
    }

*/
/////////////////////////////////////////////////
////////////  IMPRIMIR MATRIZ    ////////////////
/////////////////////////////////////////////////
    net(MLP,0,3,4) = 4;
    for (int i = 0; i < numCapas; i++) {

        for (int j = 0; j < /*numNeuronasPorCapa[i]*/NeuronsByLayer; j++) {
            for (int k = 0; k < /*numFilasPorCapa[i]*/outputSize; k++) {
                cout<< net(MLP, i, j, k)<<"\t";
            }
            cout<<endl;
        }
        cout << endl << endl;
    }
    cout << "Que paso aqui!" <<endl;
    cout<< net(MLP,0,3,4)<<endl;
    cout<< net(MLP,1,0,4)<<endl;

    for (int h = 0; h < HiddenLayers * NeuronsByLayer * outputSize; h++ )
    cout<< MLP[h]<<"\t";
    //Imprimir error
    for (int i = 0 ; i <= numDatosSalida; i++) {
        cout<<error[i]<<endl;
    }
}

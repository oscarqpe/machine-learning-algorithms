//
//  main.cpp
//  PunterosVoid RedNeuronal
//
//  Created by Andre Valdivia on 12/04/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

#include <iostream>
#include <math.h>

using namespace std;

int main(int argc, const char * argv[]) {
    
/////////////////////////////////////////////////
///////////   INICIALIZACION DE MLP   ///////////
/////////////////////////////////////////////////
    
    double e = exp(1);
    
    int numDatosEntrada = 2;
    int numDatosSalida = 2;
    int numCapas;
    
    double real[numDatosSalida+1];
    double error[numDatosSalida+1];
    
    cout<<"Ingrese el numero de capas intermedias: ";
    cin >> numCapas;
    numCapas += 2;
    int numNeuronasPorCapa[numCapas];
    int numNeuronasPorCapaAnterior[numCapas + 1];
    double*** MLP;
    MLP = (double***)malloc(sizeof(double)*numCapas);
//Inicializar numCapas
    
    //Capa de entrada. Numero de datos de entrada mas el bias
    MLP[0] = (double**)malloc(sizeof(double)*numDatosEntrada + 1);
    numNeuronasPorCapa[0] = numDatosEntrada + 1;
    for (int i = 0; i < numNeuronasPorCapa[0]; i++) {
        // 3 Filas por neurona
        MLP[0][i] = (double*)malloc(sizeof(double)*3);
        for (int j = 0; j < 3; j++) {
            MLP[0][i][j] = 0;
        }
    }
    
    //Capas intermedias
    for (int i = 1; i<numCapas-1; i++) {
        int numNeuronas;
        cout<<"Ingrese el numero de neuronas en la capa " << i <<": ";
        cin >> numNeuronas;
        numNeuronas += 1;
        numNeuronasPorCapa[i] = numNeuronas;
        //Neuronas por capa, ya se le sumo una del bias
        MLP[i] = (double**)malloc(sizeof(double)*numNeuronas);
        for (int j = 0; j < numNeuronas; j++) {
            //Filas por neurona por capa mas 3 del net, out y delta
            MLP[i][j] = (double *)malloc(sizeof(double)*(numNeuronasPorCapa[i-1] + 3));
            for (int k = 0; k < numNeuronasPorCapa[i-1] + 3; k++) {
//                cout<<"Num neuronas anterior: "<<numNeuronasPorCapa[i]<<" "<< i << "-"<<j<<"-"<<k<<endl;
                MLP[i][j][k] = 0;
            }
        }
    }
    
    //Capa de salida
    numNeuronasPorCapa[numCapas-1] = numDatosSalida + 1;
    MLP[numCapas-1] = (double **)malloc(sizeof(double)*numNeuronasPorCapa[numCapas-1]);
    for (int i = 0; i < numNeuronasPorCapa[numCapas-1]; i++) {
        MLP[numCapas-1][i] = (double *)malloc(sizeof(double) * (numNeuronasPorCapa[numCapas-2]+3));
        for (int j = 0; j < numNeuronasPorCapa[numCapas-2] + 3; j++) {
            MLP[numCapas-1][i][j] = 0;
//            cout<<"Num neuronas anterior: "<<numNeuronasPorCapa[numCapas-1]<<" "<< numCapas-1 << "-"<<i<<"-"<<j<<endl;
        }
    }
    
    //TRUQUITO JIJI
    numNeuronasPorCapaAnterior[0] = 0;
    for (int i = 1 ; i < numCapas + 1; i++) {
        numNeuronasPorCapaAnterior[i] = numNeuronasPorCapa[i-1];
    }
    
    int numFilasPorCapa[numCapas];
    for (int i = 0 ; i < numCapas; i++) {
        numFilasPorCapa[i] = numNeuronasPorCapaAnterior[i] + 3;
    }
    
/////////////////////////////////////////////////
/////////////  SETEAR MATRIZ    /////////////////
/////////////////////////////////////////////////
    
//    Inicializar bias
    for (int i = 0 ; i < numCapas; i++) {
        MLP[i][0][numNeuronasPorCapaAnterior[i] + 1] = 1;
    }
    
    //Setear la matriz de ejemplo
    MLP[0][1][1] = 0.05;
    MLP[0][2][1] = 0.10;
    
    MLP[1][1][0] = 0.35;//b1
    MLP[1][2][0] = 0.35;//b1
    MLP[1][1][1] = 0.15;//w1
    MLP[1][1][2] = 0.20;//w2
    MLP[1][2][1] = 0.25;//w3
    MLP[1][2][2] = 0.30;//w4
    
    MLP[2][1][0] = 0.60;//b2
    MLP[2][2][0] = 0.60;//b2
    MLP[2][1][1] = 0.40;//w5
    MLP[2][1][2] = 0.45;//w6
    MLP[2][2][1] = 0.50;//w7
    MLP[2][2][2] = 0.55;//w8
    
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
    
    for (int i = 1; i < numCapas; i++) {
        for (int j = 1; j < numNeuronasPorCapa[i]; j++) {
            
            MLP[i][j][numFilasPorCapa[i]-3] = 0 ;//Resetear NET
            
            for (int k = 0; k < numFilasPorCapa[i] -3 ; k++) {
                MLP[i][j][numFilasPorCapa[i]-3] += MLP[i-1][k][numFilasPorCapa[i-1]-2] * MLP[i][j][k];//NET
            }
            MLP[i][j][numFilasPorCapa[i]-2] = 1/(1 + pow(e, -MLP[i][j][numFilasPorCapa[i]-3]));//OUT
        }
    }
    
/////////////////////////////////////////////////
////////////////  ERROR    //////////////////////
/////////////////////////////////////////////////
    
    error[0] = 0;
    for (int i = 1; i <= numDatosSalida; i++) {
        error[i] = pow(real[i] - MLP[numCapas-1][i][numFilasPorCapa[numCapas-1]-2], 2)/2 ;
        error[0] += error[i];
    }

/////////////////////////////////////////////////
////////////////  BACKWARD    ///////////////////
/////////////////////////////////////////////////
    
    
    
    
/////////////////////////////////////////////////
////////////  IMPRIMIR MATRIZ    ////////////////
/////////////////////////////////////////////////

    for (int i = 0; i < numCapas; i++) {
        for (int j = 0; j < numNeuronasPorCapa[i]; j++) {
            for (int k = 0; k < numNeuronasPorCapaAnterior[i] + 3; k++) {
                cout<< MLP[i][j][k]<<" - ";
            }
            cout<<endl;
        }
        cout << endl << endl;
    }
    cout << "Que paso aqui!" <<endl;
    
    for (int i = 0 ; i <= numDatosSalida; i++) {
        cout<<error[i]<<endl;
    }
}

//
//  main.cpp
//  MLP Cuda
//
//  Created by Andre Valdivia on 19/04/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//


#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>



#define numSalidas 8
#define numEntradas 58
#define numCapas 2
#define lRate 0.3
#define numMaxNeuronas 58
#define numMaxColumnas (numMaxNeuronas + 1)
#define numData 609

#define euler exp(1)

using namespace std;

double rand0to1(){
    return rand()/double(RAND_MAX);
}

double* Data(double*_data, int fila, int columna ){
    const int pos = (columna + fila* (numEntradas + 1 ));
    return &_data[pos];
}
double* MLP(double* _MLP, int capa, int neurona, int columna){
    const int pos = (columna + (numMaxColumnas * neurona))+(capa * numMaxColumnas * numMaxNeuronas);
    return &_MLP[pos];
}

double* OutDelta(double* _out,int capa, int neurona){
    const int pos = neurona + capa*(numMaxNeuronas + 1);
    return &_out[pos];
}

void init(double* _MLP, int* numNeuronasPorCapa, int* numColumnasPorCapa, double* _out, double* _delta, double* entrada,double* real){
    
    //Init numNeuronasPorCapa
    for (int i = 0; i < numCapas - 1; i++) {
        printf("Ingrese el numero de neuronas: ");
        scanf("%d", &numNeuronasPorCapa[i]);
//        cout<< "Ingrese las neuronas en la capa "<< i << " : ";
//        cin >> numNeuronasPorCapa[i];
    }
    numNeuronasPorCapa[numCapas - 1] = numSalidas;
    
    //Init numColumnasPorCapa
    numColumnasPorCapa[0] = numEntradas + 1;
    for (int i = 1; i < numCapas; i++) {
        numColumnasPorCapa[i] = numNeuronasPorCapa[i-1] + 1;
    }
    
    //Init MLP
    int max = numCapas * numMaxNeuronas * numMaxColumnas;
    for (int i = 0; i < max; i++) {
        //        _MLP[i] = 0;
        _MLP[i] = rand0to1();
        //        _MLP[i] = 0.5;
    }
    
    //Init out y delta
    max = (numMaxNeuronas + 1) * numCapas;
    for (int i = 0; i < max; i ++) {
        _out[i] = 0;
        _delta[i] = 0;
    }
    
    for (int i = 0; i < numCapas; i++) {
        *OutDelta(_out, i, 0) = 1;
    }
    //Init entrada
    entrada[0] = 1;
    
    //Init real
    real[0] = 0;
    
}

void imprimirHeader(int* numNeuronasPorCapa, int* numColumnasPorCapa){
    //    cout<< endl;
    printf("\n numNeuronasPorCapa: \n");
    //    cout<< "numNeuronasPorCapa:"<<endl;
    for (int i = 0; i < numCapas; i++) {
        //        cout << numNeuronasPorCapa[i] << "\t";
        printf("%d\t",numNeuronasPorCapa[i]);
    }
    printf("\n\n numColumnasPorCapa\n");
    //    cout<<endl<<endl;
    
    //    cout<< "numColumnasPorCapa:"<<endl;
    for (int i = 0; i < numCapas; i++) {
        //        cout << numColumnasPorCapa[i] << "\t";
        printf("%d\t",numColumnasPorCapa[i]);
    }
    //    cout<<endl<<endl;
    printf("\n\n");
}

void imprimir_MLP(double* _MLP, int* numNeuronasPorCapa, int* numColumnasPorCapa, double* entrada, double* real, double* error, double* _out, double* _delta){
    //    cout<<endl<<"-------------- Imprimiendo matriz --------------"<<endl;
    //    cout<< "_MLP: "<<endl;
    printf("\n-----------Imprimiendo matriz -----------");
    printf("_MLP: \n");
    for (int i = 0; i < numCapas; i++) {
        //        cout<< "-- Capa "<<i<<" --"<<endl;
        printf("-- Capa %d --",i);
        for (int j = 0; j < numNeuronasPorCapa[i]; j++) {
            for (int k = 0; k < numColumnasPorCapa[i]; k++) {
                //                cout << *MLP(_MLP, i, j, k) <<"\t";
                printf("%f \t",*MLP(_MLP, i, j, k));
            }
            //            cout<< endl;
            printf("\n");
        }
        //        cout<< endl;
        printf("\n");
    }
    
    //    cout<< "---- Out ----" <<endl;
    printf("----- Out -----\n");
    for (int i = 0; i < numCapas; i++) {
        for (int j = 0; j < numNeuronasPorCapa[i] + 1; j++) {
            //            cout<< *OutDelta(_out, i, j)<< "\t";
            printf("%f \t",*OutDelta(_out, i, j));
        }
        printf("\n");
    }
    
    //    cout<<endl<<endl<< "---- Delta ----" <<endl;
    printf("\n\n----- Delta -----\n");
    for (int i = 0; i < numCapas; i++) {
        for (int j = 0; j < numNeuronasPorCapa[i] + 1; j++) {
            //            cout<< *OutDelta(_delta, i, j)<< "\t";
            printf("%f \t",*OutDelta(_delta, i, j));
        }
        //        cout<< endl;
        printf("\n");
    }
    
    //    cout<<endl<<endl<< "---- Entrada ----" <<endl;
    printf("\n\n----- Entrada -----\n");
    for (int i = 0; i < numEntradas + 1; i++) {
        //        cout<< entrada[i] << "\t";
        printf("%f\t",entrada[i]);
    }
    
    //    cout<<endl<<endl<< "---- Real ----" <<endl;
    printf("\n\n----- Real -----\n");
    for (int i = 0; i < numSalidas + 1; i++) {
        //        cout<< real[i] << "\t";
        printf("%f\t",real[i]);
    }
    
    //    cout<<endl<<endl<< "---- Error ----" <<endl;
    printf("\n\n----- Error -----\n");
    for (int i = 0; i < numSalidas + 1; i++) {
        //        cout<< error[i] << "\t";
        printf("%f\t",error[i]);
    }
}

void insertarEjemplo(double* _MLP, double* entrada, double* real){
    
    entrada[1] = 0.05;
    entrada[2] = 0.1;
    *MLP(_MLP, 0, 0, 0) = 0.35;
    *MLP(_MLP, 0, 0, 1) = 0.15;
    *MLP(_MLP, 0, 0, 2) = 0.20;
    *MLP(_MLP, 0, 1, 0) = 0.35;
    *MLP(_MLP, 0, 1, 1) = 0.25;
    *MLP(_MLP, 0, 1, 2) = 0.30;
    
    *MLP(_MLP, 1, 0, 0) = 0.6;
    *MLP(_MLP, 1, 0, 1) = 0.40;
    *MLP(_MLP, 1, 0, 2) = 0.45;
    *MLP(_MLP, 1, 1, 0) = 0.6;
    *MLP(_MLP, 1, 1, 1) = 0.50;
    *MLP(_MLP, 1, 1, 2) = 0.55;
    
    real[1] = 0.01;
    real[2] = 0.99;
    
}

double actFunct(double x){
    return 1/(1 + pow(euler, -x));
}

void forwardPropagation(double* _MLP, int* numNeuronasPorCapa, int* numColumnasPorCapa, double* _out, double* entrada){
    
    
    //Resetear outs
    for (int i = 0; i < numCapas; i++) {
        for (int j = 1; j <=numMaxNeuronas; j++) {
            *OutDelta(_out, i, j) = 0;
        }
    }
    //Forward capa 0
    for (int i = 0; i < numNeuronasPorCapa[0]; i++) {
        for (int j = 0; j < numColumnasPorCapa[0]; j++) {
            double a = entrada[j];
            double b = *MLP(_MLP, 0, i, j);
            *OutDelta(_out, 0, i + 1) +=  a * b;
        }
        *OutDelta(_out, 0, i + 1) = actFunct(*OutDelta(_out, 0, i + 1));
    }
    //Forward otras capas
    for (int i = 1; i < numCapas; i++) {
        for (int j = 0; j < numNeuronasPorCapa[i]; j++) {
            for (int k = 0; k < numColumnasPorCapa[i]; k++) {
                *OutDelta(_out, i, j+1) += *OutDelta(_out, i-1, k) * *MLP(_MLP, i, j, k);
            }
            *OutDelta(_out, i, j + 1) = actFunct(*OutDelta(_out, i, j + 1));
        }
    }
}

void backPropagation(double* _MLP, double* _out, double* _delta, int* numNeuronasPorCapa,int* numColumnasPorCapa, double* entrada, double* error,double* real){
    
    error[0] = 0;
    for (int i = 1; i <= numSalidas; i++) {
        error[i] = pow(real[i] - *OutDelta(_out, numCapas -1 , i), 2)/2 ;
        error[0] += error[i];
    }
    
    //BackPropagation
    
    //Delta 1ra capa
    for (int i = 1; i <= numNeuronasPorCapa[numCapas-1]; i++) {
        //        double outTmp = *OutDelta(_out, numCapas - 1, i);
        //        OutDelta(delta, numCapas-1, i) = (-(real[i] - outTmp))*(outTmp*(1 - outTmp)) ;
        *OutDelta(_delta, numCapas-1, i) = (-(real[i] - *OutDelta(_out, numCapas - 1, i)))*(*OutDelta(_out, numCapas - 1, i)*(1 - *OutDelta(_out, numCapas - 1, i))) ;
    }
    
    //Otros delta
    for (int i = numCapas - 2; i >= 0; i--) {
        for (int j = 1; j <= numNeuronasPorCapa[i]; j++) {
            double a = *OutDelta(_out, i, j) * (1 - *OutDelta(_out, i, j));//
            double b = 0;
            for (int k = 0; k < numColumnasPorCapa[i+1]; k++) {
                //                cout<< "K: " << k << "   num: "<< numColumnasPorCapa[i+1]<<endl;
                //                cout<< *MLP(_MLP, i+1, k, j) << " * " << *OutDelta(_delta, i+1, k+1) << endl;
                b += *MLP(_MLP, i+1, k, j) * *OutDelta(_delta, i+1, k+1);
            }
            *OutDelta(_delta, i, j) = a * b;
        }
    }
    
    //Actalizar primera capa de pesos
    for (int j = 0; j < numNeuronasPorCapa[0]; j++) {
        for (int k = 0; k < numColumnasPorCapa[0]; k++) {
            *MLP(_MLP, 0, j, k) = *MLP(_MLP, 0, j, k) - lRate * entrada[k] * *OutDelta(_delta, 0, j + 1);
        }
    }
    
    //Actualizar otras capas de pesos
    for (int i = 1; i < numCapas; i++) {
        for (int j = 0; j < numNeuronasPorCapa[0]; j++) {
            for (int k = 0; k < numColumnasPorCapa[0]; k++) {
                *MLP(_MLP, i, j, k) = *MLP(_MLP, i, j, k) - lRate * *OutDelta(_out, i-1, k) * *OutDelta(_delta, i, j + 1);
            }
        }
    }
}

void Train(double* _MLP,int* numNeuronasPorCapa, int* numColumnasPorCapa, double* _out,double* entrada, double* _delta,double* error, double* real){
    
    forwardPropagation(_MLP, numNeuronasPorCapa, numColumnasPorCapa, _out, entrada);
    backPropagation(_MLP, _out, _delta, numNeuronasPorCapa, numColumnasPorCapa, entrada, error, real);
}

int Test(double* _MLP,int* numNeuronasPorCapa, int* numColumnasPorCapa, double* _out,double* entrada, double* real, int posRespuesta){
    
    forwardPropagation(_MLP, numNeuronasPorCapa, numColumnasPorCapa, _out, entrada);
    double max = 0;
    int posMax = 0;
    for (int i = 1; i <= numSalidas; i++) {
        if (*OutDelta(_out, numCapas-1, i) > max) {
            max = *OutDelta(_out, numCapas-1, i);
            posMax = i;
        }
    }
    return posMax;
}

bool readDataIris(double* _data,int numTrain,int* TrainNumbers,int*TestNumbers){
    char buffer[4096] ;
    int ii=0,jj=0;
    // FILE *fstream = fopen("iris.txt","r");
    /Users/Andre/Dropbox/Citec/Circulo Investigacion/BasesDeDatos/cara.csv
    FILE *fstream = fopen("/Users/Andre/Dropbox/Citec/Circulo Investigacion/BasesDeDatos/iris.txt","r");
    if(fstream == NULL)
    {
        printf("\n file training opening failed ");
        return false;
    }
    while (fgets(buffer, 4096, fstream)){
        char* tmp = strdup(buffer);
        const char* tok;
        jj = 0;
        for (tok = strtok(tmp, ","); tok && *tok; tok = strtok(NULL, ",\n")){
            double i = atof(tok);
            if (i == 0) {
                if (strncmp (tok,"Iris-setosa",9) == 0) {
                    *Data(_data, ii, jj) = 1;
                }else if (strncmp (tok,"Iris-versicolor",9) == 0) {
                    *Data(_data, ii, jj) = 2;
                }else if (strncmp (tok,"Iris-virginica",9) == 0) {
                    *Data(_data, ii, jj) = 3;
                }
            }else{
                *Data(_data, ii, jj) = atof(tok);
            }
            jj++;
        }
        free(tmp);
        ii++;
    }
    
    bool seleccionados[numData];
    for (int i = 0; i < numData; i++) {
        seleccionados[i] = false;
    }
    int count = 0;
    
    while (true) {
        int s = rand () % numData;//Entre M y N
        if (seleccionados[s] == false){
            count++;
            seleccionados[s] = true;
        }
        if (count >= numTrain) {
            break;
        }
    }
    for (int i = 0, j = 0, k = 0; i < numData; i++) {
        if (seleccionados[i] == true) {
            TrainNumbers[j] = i;
            j++;
        }else{
            TestNumbers[k] = i;
            k++;
        }
    }
    return true  ;
}

bool readDataCara(double* _data,int numTrain,int* TrainNumbers,int*TestNumbers){
    char buffer[4096] ;
    int ii=0,jj=0;
    
    FILE *fstream = fopen("/Users/Andre/Dropbox/Citec/Circulo Investigacion/BasesDeDatos/cara.csv","r");
    // FILE *fstream = fopen("cara.csv","r");
    if(fstream == NULL)
    {
        printf("\n file training opening failed ");
        return false;
    }
    char* tmp2 = strdup(buffer);
    while (fgets(buffer, 4096, fstream)){
        char* tmp = strdup(buffer);
        const char* tok;
        jj = 0;
        for (tok = strtok(tmp, ";"); tok && *tok; tok = strtok(NULL, ";\n" )){
            double i = atof(tok);
            if (i == 0) {
                //                cout<<ii<<"\t"<<jj<<"\t Hay string: "<<tok<<endl;//Borrar
                *Data(_data, ii, jj) = atof(tok);
            }else{
                *Data(_data, ii, jj) = atof(tok);
            }
            jj++;
        }
        free(tmp);
        ii++;
    }
    free(tmp2);
    
    bool seleccionados[numData];
    for (int i = 0; i < numData; i++) {
        seleccionados[i] = false;
    }
    int count = 0;
    
    while (true) {
        int s = rand () % numData;//Entre M y N
        if (seleccionados[s] == false){
            count++;
            seleccionados[s] = true;
        }
        if (count >= numTrain) {
            break;
        }
    }
    for (int i = 0, j = 0, k = 0; i < numData; i++) {
        if (seleccionados[i] == true) {
            TrainNumbers[j] = i;
            j++;
        }else{
            TestNumbers[k] = i;
            k++;
        }
    }
    return true  ;
}

int main(int argc, const char * argv[]) {
    cout.precision(4);
    srand(time(NULL));
    double _MLP[numCapas * numMaxNeuronas * numMaxColumnas];
    double real[numSalidas + 1];
    double error[numSalidas + 1];
    double entrada[numEntradas + 1];
    double _out[(numMaxNeuronas + 1) * numCapas];
    double _delta[(numMaxNeuronas + 1) * numCapas];
    
    
    double* _data = (double*)malloc(sizeof(double)*numData*(numEntradas+1));
    //    double data[numData][numEntradas + 3];
    
    int numNeuronasPorCapa[numCapas];
    int numColumnasPorCapa[numCapas];
    init(_MLP, numNeuronasPorCapa, numColumnasPorCapa, _out, _delta, entrada,real);
    imprimirHeader(numNeuronasPorCapa,numColumnasPorCapa);
    
    //Leer data
    const int numTrain = numData/2; //Cambiar aqui para el porcentaje de entrenamiento
    const int numTest = numData - numTrain;
    int TrainNumbers[numTrain];
    int TestNumbers[numTest];
    
    
    //    if(readDataIris(_data, numTrain, TrainNumbers, TestNumbers) == false){
    //        return -1;
    //    }
    
    if(readDataCara(_data, numTrain, TrainNumbers, TestNumbers) == false){
        return -1;
    }
    
    //    cout<< "------ Imprimiendo Data -----------"<<endl;
    //    for (int i = 0; i < numData; i++) {
    //        for (int j = 0; j < numEntradas; j++) {
    //            cout<< *Data(_data, i, j) << "\t";
    //        }
    //        cout<<endl;
    //    }
    //    cout<<endl<<endl;
    
    //Entrenar
    int posClase = 0;
    
    for (int i = 0; i < 10000; i++) {
        for(int j = 0;j < numTrain;j++){
            for (int k = 1; k <= numEntradas; k++) {
                entrada[k] = *Data(_data, TrainNumbers[j], k);
            }
            int respuesta = *Data(_data, TrainNumbers[j], posClase);
            for (int k = 1; k <= numSalidas; k++) {
                real[k] = 0.01;
            }
            real[respuesta] = 0.99;
            Train(_MLP, numNeuronasPorCapa, numColumnasPorCapa, _out, entrada, _delta, error, real);
            //            imprimir_MLP(_MLP, numNeuronasPorCapa, numColumnasPorCapa, entrada, real, error, _out, _delta);
        }
    }
    
    //Testear
    int correctas = 0;
    int incorrectas = 0;
    int incorrectasPorDato[numSalidas + 1];
    for (int i = 1; i<=numSalidas; i++) {
        incorrectasPorDato[i] = 0;
    }
    for(int j = 0;j < numTest;j++){
        for (int k = 1; k <= numEntradas; k++) {
            entrada[k] = *Data(_data, TestNumbers[j], k);
        }
        int respuesta = *Data(_data, TestNumbers[j], posClase);
        for (int k = 1; k <= numSalidas; k++) {
            real[k] = 0.01;
        }
        real[respuesta] = 0.99;
        int resp = Test(_MLP, numNeuronasPorCapa, numColumnasPorCapa, _out, entrada, real, respuesta);
        //        cout<<entrada[0]<<"\t"<<entrada[1]<<"\t"<<entrada[2]<<"\t"<<entrada[3]<<"\t"<<entrada[4]<<endl;
        //        cout<<real[0]<<"\t"<<real[1]<<"\t"<<real[2]<<"\t"<<real[3]<<"\t"<<real[4]<<endl;
        //        cout<< data[TestNumbers[j]][0]<< "\t" << data[TestNumbers[j]][1]<< "\t"<< data[TestNumbers[j]][2] << "\t"<< data[TestNumbers[j]][3]<< "\t"<< data[TestNumbers[j]][4]<<endl;
        //        cout<<"Resp: "<< resp <<endl;
        //        cout<<"Respuesta real: "<< respuesta <<endl<<endl;
        if (resp == respuesta) {
            correctas++;
        }else{
            incorrectasPorDato[respuesta]++;
            incorrectas++;
        }
        //            imprimir_MLP(_MLP, numNeuronasPorCapa, numColumnasPorCapa, entrada, real, error, _out, _delta);
    }
    cout<<endl<<endl;
    cout<<"Correctas: "<< correctas<<endl;
    cout<<"Incorrectas: "<< incorrectas<<endl;
    
    for (int i = 1; i<= numSalidas; i++) {
        cout<< i<<": "<< incorrectasPorDato[i]<<endl;
    }
    double porcentaje = (double)correctas/(double)numTest * 100;
    cout<< "Un porcentaje de: "<< porcentaje <<"%"<<endl;
    //    insertarEjemplo(_MLP,entrada,real);    //Funcion insertar data ejemplo
    //    forwardPropagation(_MLP, numNeuronasPorCapa, numColumnasPorCapa, _out, entrada);
    //    backPropagation(_MLP, _out, _delta, numNeuronasPorCapa, numColumnasPorCapa, entrada, error, real);
    
    //    imprimir_MLP(_MLP, numNeuronasPorCapa, numColumnasPorCapa, entrada, real, error, _out, _delta);
    free(_MLP);
    free( _out);
    free(_delta);
    free(_data);
}

//
//  main.cpp
//  mlp-array2
//
//  Created by Ricardo Coronado on 17/05/16.
//  Copyright © 2016 Ricardo Coronado. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include <random>

using namespace std;


//  ESTRUCTURA PARA DATA DE ENTRENAMIENTO Y TEST
struct _datos
{
    double * data = NULL;
    int target;
    struct _datos *sig = NULL;
};

//  FUNCIONES DEL PROYECTO
void mlp_fordward();
void mlp_backward(_datos *data);
void generar_deltas(int pos_target,int pos, int capaAux);
void multiplicacion_punto(int pos_input,int pos_peso,int cant_input, int cant_nueronas);
void actualizar_pesos(int posInput, int posDelta, int cant_input, int cant_deltas,int numeroCapa);
double f_signoid(double numero);

//  Auxiliares
void obtener_data(string archivo,int total_input);
void copiar_input(_datos *origen, int cantidad);
void rand_distribucion_normal(double *capa, int inicio, int num);
//void rand_cero_uno(long double *capa, int inicio, int num);

//  Para imprimir Data
void imprimir_input_mlp(int posInput , int total_salidas, int numCapa);
void imprimir_filtro_mlp(int pos1, int pos2 , int * neurona);

//  Otros
void liberar_memoria();
void detalles_test(int contador_casos_entrenamiento,int contadorCasos, int contador, int clasificados);
void matriz_confusion();


// VARIABLES
_datos *point_data; // Puntero a la data cargada
int total_capas_mlp; // # de capas mlp
int *pos_mlp; // posiciones iniciales de cada capa
int total_array_mlp; // # tamaño total del arreglo
double *capa_mlp;
double *point_capa_mlp; // Puntero auxiliar al array capa_mlp
double **sumatorias_error; // error acumulado para cada neurona
int total_datos;

//  Clock
clock_t start;
double duration;

//  Codificacion de Targets
double target[10][10] = {{0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01,0.01},
    {0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01,0.01},
    {0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01,0.01,0.01},
    {0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01,0.01,0.01},
    {0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01,0.01,0.01},
    {0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01,0.01,0.01},
    {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01,0.01,0.01},
    {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99,0.01,0.01},
    {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.99,0.01},
    {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01,0.99}};

//  Matriz de confusion
double mConf[10][10] = {{0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0}};


//  Arquitectura de MLP
int neurona[4] = {64,10,10,0};

//  Tasas de aprendizaje
double learn_rate_mlp = 0.7; // 0.01

//  Epocas de entrenamiento
int total_epocas = 400;

//  Dataset
//char *rutaDataTrain = "mnist_train_28x28.csv";
//char *rutaDataTest = "mnist_test_28x28.csv";
char *rutaDataTrain = "mnist.csv";
char *rutaDataTest = "mnist_test.csv";


int main(int argc, const char * argv[]) {
    
//  ------------------------------------------------------------------------------------------------------------------------
//  ESTRUCTURA ARRAY MLP ---------------------------------------------------------------------------------------------------
    
    
    total_capas_mlp = (sizeof(neurona)/sizeof(*neurona)) - 1;
    int total_input = neurona[0];
    int total_salidas = neurona[total_capas_mlp-1];
    
    //  Almacenamos la posicion de cada capa en el array pos_mlp
    pos_mlp = (int *)malloc(sizeof(int)*total_capas_mlp);
    int total = 0;
    for (int i = 0; i < total_capas_mlp; i++)
    {
        pos_mlp[i] = total;
        total = total + neurona[i] + 1 + (neurona[i]+1) * (neurona[i+1]);
    }
    
    //  Guardamos el tamano total del array cnn en una variable global
    total_array_mlp = total;
    
    //  Inicializamos el array con 0,Generamos los pesos aleatorio por una distribucion uniforme N(u,o2) = N(0,2/n) - n es la cantidad de pesos generados
    capa_mlp = (double *)malloc(sizeof(double)*total_array_mlp);
    point_capa_mlp = capa_mlp; // Puntero auxiliar a capa_mlp
    for (int i = 0; i < total_capas_mlp; i++)
    {
        //  valores de la neuorna inicializado en 0
        for (int j = 0; j < neurona[i]+1; j++)
        {
            capa_mlp[pos_mlp[i] + j] = 0;
        }
        // pesos inicializados
        rand_distribucion_normal(point_capa_mlp,pos_mlp[i] + neurona[i] + 1,(neurona[i]+1) * (neurona[i+1]));
//        rand_cero_uno(point_capa_mlp,pos_mlp[i] + neurona[i] + 1,(neurona[i]+1) * (neurona[i+1]));
    }
    
    //  Reservamos espacio en memoria para una matriz que almacene el error acumulado de una neurona
    sumatorias_error = (double **)malloc(sizeof(double)*(total_capas_mlp-1));
    for (int i = 0; i< total_capas_mlp-1; i++) {
        sumatorias_error[i] = (double *)malloc(sizeof(double)*(neurona[i]+1));
        for (int j=0; j < neurona[i]+1; j++) {
            sumatorias_error[i][j] = 0;
        }
    }
    
    
//  ------------------------------------------------------------------------------------------------------------------------
//  ENTRENAMIENTO DE LA RED ------------------------------------------------------------------------------------------------
    
    
    cout<<"Train..."<<endl;
    
    _datos *data; // puntero auxiliar a los datos
    
    //  Obtener la data de entrenamiento en una lista enlazada
    total_datos = 0;
    obtener_data(rutaDataTrain,total_input);
    int total_datos_entrenamiento = total_datos; // Contador de datos de entrenamiento
    
    //  Iniciamos a correr el reloj
    start = clock();
    
    //  Recorremos las epocas de entrenamiento
    for (int epoca = 0; epoca< total_epocas; epoca++)
    {
        cout<<"Epoca "<<epoca<<endl;
        //  Apuntamos al primer dato del dataset
        data = point_data;
        
        //  Recorremos la data que esta almacenada en una lista enlazada
        //for (int i = 0; i < 2; i++)
        while (data != NULL)
        {
            // Insertamos la imagen en la estructura del array ocupa las posicion [0 - total_imagen]
            copiar_input(data,total_input);
            // Fordward de MLP
            mlp_fordward();
            // Backward de MLP - recibe la data para extraer la etiqueta del caso de entrenamiento
            mlp_backward(data);
            
            // saltamos a la siguiente posicion de la lista
            data = data->sig;
        }
    }
    
    // Detenemos el reloj que mide el tiempo de entrenamiento
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    
    
//  ------------------------------------------------------------------------------------------------------------------------
//  TEST DE LA RED ---------------------------------------------------------------------------------------------------------
    
    
    cout<<"Test..."<<endl;
    
    //  Obtener la data de entrenamiento en una lista enlazada
    total_datos = 0;
    obtener_data(rutaDataTest,total_input);
    int total_datos_test = total_datos; // Contador de datos de test
    data = point_data;
    
    //  Resultados
    int pos_resultado = pos_mlp[total_capas_mlp - 1] + 1; // posicion inicial del output final
    int res;
    
    //  Contadores para el resultado final
    int contador_aciertos = 0;
    int salidaFinal = 0;
    int clasificados = 0;
    string salida;
    
    
    //for (int i = 0; i < 10000; i++)
    while (data != NULL)
    {
        res = data->target; // Almacenamos la etiqueta del caso de test
        copiar_input(data,total_input); // copiamos la imagen en el array cnn
        mlp_fordward(); // Fordward mlp
        
        
        //  Conteo de aciertos
        for (int i = 0; i < total_salidas; i++)
        {
            //  Redondeamos la salida a 0 o 1
            salidaFinal = (int)(point_capa_mlp[pos_resultado+i]+0.5);
            
            // Contador de aciertos
            if(res == i && salidaFinal == 1)
            {
                contador_aciertos++;
            }
            
            //  Conteo para Matriz de confusion
            if (salidaFinal == 1) {
                mConf[i][res] = mConf[i][res] + 1;
                clasificados++;
            }
            
            //cout<<salidaFinal<<" - ";
            //cout<<point_capa_mlp[pos_resultado+i]<<" - ";
        }
        //cout<<"- "<<res<<endl;
        
        
        // saltamos a la siguiente posicion de la lista
        data = data->sig;
    }

    //  ------------------------------------------------------------------------------------------------------------------------
    //  IMPRIMIR RESULTADOS ----------------------------------------------------------------------------------------------------
    
    detalles_test(total_datos_entrenamiento,total_datos_test, contador_aciertos, clasificados);
    matriz_confusion();
    
    //  ------------------------------------------------------------------------------------------------------------------------
    //  LIBERAR MEMORIA ----------------------------------------------------------------------------------------------------
    
    liberar_memoria();
    point_data = NULL;
    free(point_data);
    point_capa_mlp = NULL;
    free(point_capa_mlp);
    
    return 0;
}




//  ------------------------------------------------------------------------------------------------------------------------
//  DESARROLLO DE LA FUNCIONES ---------------------------------------------------------------------------------------------



//  PARA MLP ----------------------------------------------------------------------------------------------------------------

void mlp_fordward()
{
    
    //  FORWARD PROPAGATION
    for (int c = 0; c < total_capas_mlp-1; c++)
    {
        multiplicacion_punto(pos_mlp[c], pos_mlp[c+1], neurona[c], neurona[c+1]);
    }
}

void mlp_backward(_datos * data)
{
    //  BACKWARD PROPAGATION
    for (int c = total_capas_mlp-2; c > -1; c--)
    {
        if (c == total_capas_mlp-2)
        {
            generar_deltas(data->target, pos_mlp[c+1], neurona[c+1]);
        }
        actualizar_pesos(pos_mlp[c],pos_mlp[c+1],neurona[c],neurona[c+1],c);
    }
    
}

//  GENERAMOS LOS DELTAS PARA LA PRIMERA ITERACION DEL BACKWARD PROPAGATION
void generar_deltas(int pos_target,int pos, int capaAux)
{
    for (int i = 1; i < capaAux+1; i++) {
        double out = point_capa_mlp[pos+i];
        double obj = target[pos_target][i-1];
        point_capa_mlp[pos+i] = ( out - obj) * out * (1 - out);
    }
}

//  SIMULACION DE MULTIPLICACION DE MATRICES
//  pos_input -> manda la posicion inicial de las input para la capa
//  pos_peso -> manda la posicion inicial de los pesos
//  cant_input -> cantidad de Inputs que alimentan a la capa actual = cantidad de pesos que recibe la neurona de la capa actual
//  cant_neuronas -> cantidad de neuronas a procesar en la capa actual
void multiplicacion_punto(int pos_input,int pos_peso,int cant_input, int cant_nueronas)
{
    double sumatoria = 0;
    int ini_matriz = cant_input + 1;
    point_capa_mlp[pos_peso] = 1;
    
    for (int i=1; i< cant_nueronas +1; i++){
        sumatoria = 0;
        for (int j=0; j<cant_input+1; j++){
            sumatoria = sumatoria + point_capa_mlp[pos_input+j]* point_capa_mlp[pos_input+ini_matriz*i+j];
        }
        point_capa_mlp[pos_peso+i] = f_signoid(sumatoria);
    }
}

//  ACTUALIZAR PESOS
void actualizar_pesos(int posInput, int posDelta, int cant_input, int cant_deltas,int numeroCapa)
{
    // indicie para marcar el inicio de la seccion de pesos de la capa i
    int ini_matriz = posInput + cant_input + 1;
    double derivada;
    
    //  Recorremos los deltas de la capa que hallamos anteriormente - hay un delta por neurona
    for (int i=1; i < cant_deltas+1; i++)
    {
        //  Recorremos las entradas de las neuronas de la capa siguiente
        for (int j=0; j < cant_input+1; j++)
        {
            //  Acumulamos el error cometido para casa neurona j esima de la capa l
            sumatorias_error[numeroCapa][j] = sumatorias_error[numeroCapa][j] + (point_capa_mlp[posDelta+i] * point_capa_mlp[ini_matriz+j]);
            //  La derivada del error total respecto a un peso wi es igual al delta x por la entrada de la neurona relacionada al peso
            derivada = point_capa_mlp[posDelta + i] * point_capa_mlp[posInput+j];
            //  Actualizamos el peso
            point_capa_mlp[ini_matriz + j] = point_capa_mlp[ini_matriz + j] - learn_rate_mlp * derivada;
        }
        
        //  como utilizamos un array para almacenar toda la data debemos actualizar el indice para simular el comportamiento de matriz
        ini_matriz = ini_matriz + cant_input + 1;
    }
    
    //  Por generalizacion acumulamos el error del bias que se almacena en la posicion 0, pero esto no es necesario ya que el bias es una neurona aislada, la primera posicion la inicializamos a 0 para las proximas iteraciones
    double input_aux = 0;
    sumatorias_error[numeroCapa][0] = 0;
    
    if (numeroCapa > 0){
        for (int i=1; i < cant_input + 1; i++)
        {
            //  Almacenamos el delta para la iteracion de la siguiente capa, en nuestro array esto lo almacenamos en los input de salida de la proxima capa ya que estos no se volveran a utilizar, de esta manera reducimos el tamaño de nuestro array
            input_aux = point_capa_mlp[posInput + i];
            point_capa_mlp[posInput + i] = sumatorias_error[numeroCapa][i] * (input_aux * (1 - input_aux));
            //  Inicializamos el acumulador a 0 para proximas iteraciones
            sumatorias_error[numeroCapa][i] = 0;
        }
    }else{
//        //  En el caso de la primera capa el error solo depende del delta acumulado ya que no hay activacion de la neurona por ser dato de entrada
//        for (int i=1; i < cant_input + 1; i++)
//        {
//            point_capa_mlp[posInput + i] = sumatorias_error[numeroCapa][i];
//            //  Inicializamos el acumulador a 0 para proximas iteraciones
//            sumatorias_error[numeroCapa][i] = 0;
//        }
    }
}

//  FUNCION DE ACTIVACION
double f_signoid(double numero)
{
    return 1 / (1 + pow(exp(1), -1 * numero));
}



// AUXILIARES ----------------------------------------------------------------------------------------------------------------

//  GENERAMOS UNA ESTRUCTURA CON LA DATA DE ENTRENAMIENTO O TEST
void obtener_data(string archivo,int total_input){
    
    string numero;
    int a,b;
    _datos *nuevo, *puntero_aux;
    
    nuevo = (struct _datos *) malloc (sizeof(struct _datos));
    puntero_aux = nuevo;
    point_data = nuevo;
    
    
    ifstream file(archivo);
    string value;
    while (file.good())
    {
        total_datos++;
        getline ( file, value, '\n' );
        
        //  declaramos un array
        double *d_array;
        d_array = (double *)malloc(sizeof(double)*total_input);
        int c = 0;
        
        //  recorremos cada linea del file
        for (int i=0; i<total_input; i++)
        {
            a = (int)value.find(';',c);
            b = (int)value.find(';',a+1);
            c = b;
            numero = value.substr(a+1,b-a-1);
            d_array[i] = stof(numero)/255;
        }
        
        //  almacenamos los datos en la lista y enlazamos
        a = (int)value.find(';');
        numero = value.substr(0,a);
        nuevo->target = stoi(numero);
        nuevo->data = d_array;
        puntero_aux->sig = nuevo;
        nuevo->sig = NULL;
        puntero_aux = nuevo;
        
        nuevo = (struct _datos *) malloc (sizeof(struct _datos));
    }
    
    //  liberamos la memoria
    puntero_aux = NULL;
    free(nuevo);
    free(puntero_aux);
}


//  INICIALIZAMOS LAS ENTRADAS PARA LA PRIMERA CAPA DE CADA CASO DE ENTRENAMIENTO
void copiar_input(_datos *origen, int cantidad)
{
    double aux;
    point_capa_mlp[0] = 1;
    for (int i=0; i<cantidad; i++) {
        aux = origen->data[i];
        point_capa_mlp[i] = aux;
    }
}

//  NUMEROS ALEATORIOA DE DISTRIBUCION NORMAL
void rand_distribucion_normal(double *capa, int inicio, int num)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //    double* ret = (double*)malloc(sizeof(double)*num);
    default_random_engine generator(seed);
    double sigma = sqrt(double(2)/num);
    normal_distribution<double> distribution(0,sigma);
    for (int i = 0; i < num;i++){
        capa[inicio + i] = distribution(generator);
    };
}

//  NUMEROS ALEATORIO ENTRE [0 y 1]
void rand_cero_uno(double *capa, int inicio, int num)
{
    for (int i = 0; i < num;i++){
        capa[inicio + i] = (double) rand()/RAND_MAX;
        cout<<capa[inicio + i]<<endl;
    };
}


// IMPRIMIR ------------------------------------------------------------------------------------------------------------------

void imprimir_input_mlp(int posInput , int total_salidas, int numCapa)
{
    
    cout<<"Input MLP Capa "<<numCapa<<endl<<endl;
    cout<<capa_mlp[posInput];
    for (int i = 0; i < total_salidas; i++) {
        cout<<" - "<<capa_mlp[posInput + i + 1];
    }
    cout<<endl<<endl;
}

void imprimir_filtro_mlp(int pos1, int pos2 , int * neurona)
{
    int total1 = pos1 + neurona[0]+1 + (neurona[0]+1) * neurona[1];
    
    cout<<"Filtro Capa 1"<<endl<<endl;
    
    pos1 = pos1 + neurona[0]+1;
    
    for (int i = 0; i < neurona[1]; i++)
    {
        for (int j = 0; j < neurona[0]+1; j++)
        {
            cout<<capa_mlp[pos1 + j + i*(neurona[0]+1)]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    
    
    cout<<"Filtro Capa 2"<<endl<<endl;
    
    pos2 = pos2 + neurona[1]+1;
    
    for (int i = 0; i < neurona[2]; i++)
    {
        for (int j = 0; j < neurona[1]+1; j++)
        {
            cout<<capa_mlp[pos2 + j + i*(neurona[1]+1)]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}


// OTROS ------------------------------------------------------------------------------------------------------------------

//  RESULTADOS DEL TEST Y PRESICION
void detalles_test(int contador_casos_entrenamiento,int contadorCasos, int contador, int clasificados)
{
    int acurracy = contador * 100 / contadorCasos;
    int noClasificados = contadorCasos - clasificados;
    cout<<endl<<"--------------------------------"<<endl;
    cout<<"Tiempo de Entrenamiento: "<< duration <<endl;
    cout<<"Epocas: "<<total_epocas<<endl;
    cout<<"Casos Entrenamiento: "<<contador_casos_entrenamiento<<endl;
    cout<<"Learn Rate Mlp: "<<learn_rate_mlp<<endl;
    cout<<"--------------------------------"<<endl;
    cout<<"Casos de Test: "<<contadorCasos<<endl;
    cout<<"Aciertos: "<<contador<<endl;
    cout<<"Sin Clasificar: "<<noClasificados<<endl;
    cout<<"Desaciertos: "<<contadorCasos - contador - noClasificados<<endl;
    cout<<"Accuracy: "<<acurracy<<"%"<<endl;
}


//  MATRIZ DE CONFUSION
void matriz_confusion()
{
    cout<<endl;
    cout<<"MATRIZ DE CONFUSION"<<endl<<endl;
    for (int i = 0; i < neurona[2]; i++)
    {
        int sum = 0;
        cout<<"     "<<i<<" | ";
        for (int j = 0; j < neurona[2]; j++) {
            cout<<mConf[i][j]<<"  ";
            sum = sum + mConf[i][j];
        }
        cout<<" : "<<sum<<endl;
    }
    cout<<endl;
}


// LIBERAR MEMORIA
void liberar_memoria()
{
    _datos *puntero_aux,*puntero_aux2;
    puntero_aux = point_data;
    puntero_aux2 = point_data;
    while (puntero_aux2 != NULL)
    {
        puntero_aux = puntero_aux2;
        puntero_aux2 = puntero_aux2->sig;
        free(puntero_aux->data);
        free(puntero_aux);
    }
    
    puntero_aux = NULL;
    puntero_aux2 = NULL;
    free(puntero_aux);
    free(puntero_aux2);
}







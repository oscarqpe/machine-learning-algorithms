//
//  main.cpp
//  cnn
//
//  Created by Ricardo Coronado on 9/05/16.
//  Copyright © 2016 Ricardo Coronado. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include <vector>

using namespace std;

//  ESTRUCTURA PARA DATA DE ENTRENAMIENTO Y TEST

struct _datos
{
    double * data = NULL;
    int target;
    struct _datos *sig = NULL;
};

// FUNCIONES
void convolucion_pool(int * imagen, int * filtro, int *pool ,int pos_img , int pos_convol , int pos_pool, int indice_data);
void obtener_data(string archivo,int total_input, int * dim_imagen, double valor_normalizador);
void copiar_input(_datos *origen, int cantidad);
void imprimir_convolucion(int *imagen, int *filtro, int pos_convol);
void imprimir_filtros(int *imagen, int *filtro, int pos_imagen);
void imprimir_imagen(int *imagen, int pos_imagen);
void imprimir_pool(int *imagen, int *filtro, int *pool, int pos_pool);
void imprimir_input_mlp(int posInput , int total_salidas, int numCapa);
void imprimir_filtro_mlp(int pos1, int pos2 , int * neurona);
void fully_conected(int pos_pool, int total_datos);
void mlp_fordward();
void mlp_backward(_datos *data);
void generar_deltas(int pos_target,int pos, int capaAux);
void actualizar_pesos(int posInput, int posDelta, int cant_input, int cant_deltas,int numeroCapa);
double f_signoid(double numero);
void back_pool(int * imagen, int * filtro, int *pool ,int pos_img , int pos_convol , int pos_pool, int pos_delta);
void actualizar_filtros(int * imagen, int * filtro, int *pool ,int pos_img , int pos_convol , int pos_pool);
void multiplicacion_punto(int pos_input,int pos_peso,int cant_input, int cant_nueronas);
void detalles_test(int contadorCasos, int contador, int clasificados);
void liberar_memoria();
void matriz_confusion();

// VARIABLES
_datos *point_data;
int ** network;
int total_capas_cnn;
int total_array_cnn;
int *pos_cnn;
double *point_capa_cnn;
double *capa_cnn;

double learn_rate;
int *neurona;
double *capa_mlp;
double *point_capa_mlp;
double **sumatorias_error;
int total_input;
int total_salidas;
int total_capas_mlp;
int *pos_mlp;
int total_array_mlp;

int total_epocas;
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

//  VARIABLES PARA EL CLOCK

clock_t start;
double duration;

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n"<<endl;
    double total = 0;
    
//  CONFIGURACION CNN
    
    network = (int **)malloc(sizeof(int)*3);
    network[0] = (int *)malloc(sizeof(int)*4);
    network[1] = (int *)malloc(sizeof(int)*5);
    network[2] = (int *)malloc(sizeof(int)*5);
    
    network[0] = (int[4]){8,8,1,1};  // Dim Imagen [0,1,2] - Padding [3]
    network[1] = (int[5]){3,3,1,1,7};// Dim filtro [0,1,2] - Salto [3] - Filtros [4]
    network[2] = (int[5]){2,2,1,2,network[1][4]};// Dim pool [0,1,2] - Salto [3] - Filtros o profundida [4]

    total_capas_cnn = 3;
    pos_cnn = (int *)malloc(sizeof(int)*total_capas_cnn);
    
    int pad = 2 * network[0][3];
    int dimRes = network[0][0] + pad;
    int filtro = 0;
    int saltos = 0;
    int total_imagen  = dimRes * dimRes * network[0][2];
    int total_filtros = network[1][0] * network[1][1] *  network[1][2] * network[1][4];
    
    pos_cnn[0] = 0;
    total = total_imagen + total_filtros;
    for (int i = 1; i< total_capas_cnn; i++)
    {
        pos_cnn[i] = total;
        filtro = network[i][0];
        saltos = network[i][3];
        dimRes = (dimRes - filtro)/saltos + 1;
        total = total + dimRes * dimRes * network[i][4];
    }
    
    //  inicializamos el array principal de elementos con valores en cero
    total_array_cnn = total;
    capa_cnn = (double *)malloc(sizeof(double)*total_array_cnn);
    for (int i = 0; i< total_array_cnn; i++) {
        capa_cnn[i] = 0;
    }
    
    //  Inicializamos los pesos con datso aleatorios entre [0,1]
    srand(time(0));
    for (int i = total_imagen; i< total_imagen + total_filtros; i++) {
        capa_cnn[i] = (double) rand()/RAND_MAX;
    }
    
    //  Inicializamos la capa de pool con valores muy negativos para utilizar la funcion maximo
    for (int i = pos_cnn[2]; i < total_array_cnn; i++) {
        capa_cnn[i] = -10000;
    }
    

//  CONFIGURACION MLP
    
    total_capas_mlp = 3;
    neurona = (int *)malloc(sizeof(int)*total_capas_mlp+1);
    
    //  Estructura de la neurona el indice del array neurona es el numero de capas y el valor es la cantidad de neuronas
    //    neurona[0] = 58; // capa entrada
    neurona[0] = total_array_cnn - pos_cnn[2]; // capa entrada
    neurona[1] = 15;  // capa Intermedia
    neurona[2] = 10;  // capa de salida
    neurona[3] = 0;  // capa auxiliar [siempre cero]
    
    total_input = neurona[0];
    total_salidas = neurona[total_capas_mlp-1];
    learn_rate = 0.8;
    
    //  Array para almacenar posiciones de inicio de las capas
    pos_mlp = (int *)malloc(sizeof(int)*total_capas_mlp);

    //  inicializamos las posiciones de las capas - y hallamos el total de elementos del array
    total = 0;
    for (int i = 0; i < total_capas_mlp; i++)
    {
        pos_mlp[i] = total;
        total = total + neurona[i] + 1 + (neurona[i]+1) * (neurona[i+1]);
    }
    
    //  inicializamos el array principal de elementos con valores random
    total_array_mlp = total;
    srand(time(0));
    capa_mlp = (double *)malloc(sizeof(double)*total_array_mlp);
    for (int i = 0; i< total_array_mlp; i++)
    {
        capa_mlp[i] = (double) rand()/RAND_MAX;
    }
    
    //  Reservamos espacio en memoria para una matriz que almacene el error acumulado de una neurona
    sumatorias_error = (double **)malloc(sizeof(double)*(total_capas_mlp-1));
    for (int i = 0; i< total_capas_mlp-1; i++) {
        sumatorias_error[i] = (double *)malloc(sizeof(double)*(neurona[i]+1));
        for (int j=0; j < neurona[i]+1; j++) {
            sumatorias_error[i][j] = 0;
        }
    }

    
//  ENTRENAMIENTO
    
    cout<<"Train..."<<endl;
    _datos *data;
    obtener_data("mnist.csv", total_imagen,network[0], 255);
    point_capa_cnn = capa_cnn;
    point_capa_mlp = capa_mlp;
    total_epocas = 600;
    
    start = clock();
    
    //  Recorremos las epocas
    for (int epoca = 0; epoca< total_epocas; epoca++)
    {
        //  Recorremos la data que esta almacenada en una lista enlazada

        data = point_data;
//        cout<<"Epoca "<<epoca<<endl;
        //for (int i = 0; i < 550; i++)
        while (data != NULL)
        {

            copiar_input(data,total_imagen);
            convolucion_pool(network[0], network[1], network[2], pos_cnn[0],pos_cnn[1], pos_cnn[2], 1);
            fully_conected(pos_cnn[2], total_input);
            mlp_fordward();
            
            mlp_backward(data);
            back_pool(network[0], network[1], network[2], pos_cnn[0],pos_cnn[1], pos_cnn[2], pos_mlp[0]);
            actualizar_filtros(network[0], network[1], network[2], pos_cnn[0],pos_cnn[1], pos_cnn[2]);

            
            //  Inicializamos la seccion de la comnvolucion para la siguiente iteracion
            for (int i = pos_cnn[1]; i < pos_cnn[2]; i++) {
                capa_cnn[i] = 0;
            }
            
            for (int i = pos_cnn[2]; i < total_array_cnn; i++) {
                capa_cnn[i] = -10000;
            }
            
            
            data = data->sig;
        }
    }
    
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    
    
 
// TESTING
    
    cout<<"Test..."<<endl;
    
    obtener_data("mnist_test.csv",total_imagen,network[0], 255);
    point_capa_cnn = capa_cnn;
    point_capa_mlp = capa_mlp;
    data = point_data;
    
    int res;
    int pos_resultado = pos_mlp[total_capas_mlp - 1] + 1;
    string salida;
    
    int contador = 0;
    int salidaFinal = 0;
    int contadorCasos = 0;
    int clasificados = 0;
    

    while (data != NULL)
    {
        res = data->target;
        copiar_input(data,total_imagen);
        convolucion_pool(network[0], network[1], network[2], pos_cnn[0],pos_cnn[1], pos_cnn[2], 1);
        fully_conected(pos_cnn[2], total_input);
        mlp_fordward();
        
        contadorCasos++;
        
        //  Conteo de aciertos
        for (int i = 0; i < total_salidas; i++)
        {
            //  Redondeamos la salida a 0 o 1
            salidaFinal = (int)(point_capa_mlp[pos_resultado+i]+0.5);
            if(res == i && salidaFinal == 1)
            {
                contador++;
            }
            
            //  Conteo para Matriz de confusion
            if (salidaFinal == 1) {
                mConf[i][res] = mConf[i][res] + 1;
                clasificados++;
            }
            
//            cout<<salidaFinal<<" - ";
        }
        
//        cout<<"- "<<res<<endl;
        
        
        //  Inicializamos la seccion de la comnvolucion para la siguiente iteracion
        for (int i = pos_cnn[1]; i < pos_cnn[2]; i++) {
            capa_cnn[i] = 0;
        }
        
        for (int i = pos_cnn[2]; i < total_array_cnn; i++) {
            capa_cnn[i] = -10000;
        }
        
        data = data->sig;
    }
    
    detalles_test(contadorCasos, contador, clasificados);
    matriz_confusion();
    
    
    //  LIBERAMOS MEMORIA
    //  _________________
    
    liberar_memoria();
    point_data = NULL;
    free(point_data);
    point_capa_mlp = NULL;
    free(point_capa_mlp);
    point_capa_cnn = NULL;
    free(point_capa_cnn);

    return 0;
}


void convolucion_pool(int * imagen, int * filtro, int *pool ,int pos_img , int pos_convol , int pos_pool, int indice_data)
{
    //  Datos imagen
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    int pos_imagen = pos_img; // Posicion inicial de la imagen con padding
    int pos_filtro = pos_imagen + ancho_imagen * ancho_imagen * profundidad_imagen; // Posicion inicial del filtro

    //  Datos filtro
    int num_filtros = filtro[4];
    int saltos = filtro[3];
    int ancho_filtro = filtro[0];
    int ancho_convol = (ancho_imagen - ancho_filtro) / saltos + 1;

    //  Datos Pool
    int ancho_filtro_pool = pool[0];
    int saltos_pool = pool[3];
    int ancho_pool = (ancho_convol - ancho_filtro_pool) / saltos_pool + 1;
    
    for (int fil = 0 ; fil < num_filtros; fil++)
    {
        //  Recorre el alto de la matriz de convolucion
        int xp = 0;
        for (int pi = 0; pi < ancho_convol; pi++)
        {
            //  Recorre el alto del filtro
            for (int row = 0; row < ancho_filtro; row++)
            {
                //  Recorre el ancho de la matriz de convolucion - acumula el producto
                for (int pj = 0; pj < ancho_convol; pj++)
                {
                    double sumatoria = 0;
                    
                    //  Recorre el filtro y realiza la multiplicacion
                    for (int col = 0; col < ancho_filtro; col++)
                    {
                        
                        int xpos_filtro = pos_filtro + col + (row * ancho_filtro) + (fil*ancho_filtro * ancho_filtro);
                        int xpos_imagen = pos_imagen + col + (pj * saltos) + (row * ancho_imagen) + (pi * ancho_imagen);
                        double afil = capa_cnn[xpos_filtro];
                        double aimg = capa_cnn[xpos_imagen];
                        sumatoria = sumatoria + afil * aimg;
                    
                    }
                    
                    int xpos_convol = pos_convol + pj + (pi * ancho_convol) + (fil * ancho_convol * ancho_convol);
                    capa_cnn[xpos_convol] = capa_cnn[xpos_convol] + sumatoria;
                }
            }
            
            //  Realizamos el Pooling
            if (pi != 0 && pi % ancho_filtro_pool == 0)
            {
                xp++;
            }
            
            for (int x=0; x < ancho_pool; x++)
            {
                double max = capa_cnn[pos_pool + x + (xp * ancho_pool) + (fil * ancho_pool * ancho_pool)];
                
                for (int y = 0; y < ancho_filtro_pool; y++)
                {
                    double valor = capa_cnn[pos_convol + y + (x * ancho_filtro_pool)+(pi * ancho_convol)+(fil * ancho_convol * ancho_convol)];
                    if (valor > max)
                    {
                        max = valor;
                    }
                }
                
                capa_cnn[pos_pool + x + (xp * ancho_pool) + (fil * ancho_pool * ancho_pool)] = max;
            }
        }
    }
    
}

void actualizar_filtros(int * imagen, int * filtro, int *pool ,int pos_img , int pos_convol , int pos_pool)
{
    //  Datos imagen
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    int pos_imagen = pos_img; // Posicion inicial de la imagen con padding
    int pos_filtro = pos_imagen + ancho_imagen * ancho_imagen * profundidad_imagen; // Posicion inicial del filtro
    
    //  Datos filtro
    int num_filtros = filtro[4];
    int saltos = filtro[3];
    int ancho_filtro = filtro[0];
    int ancho_convol = (ancho_imagen - ancho_filtro) / saltos + 1;
    
    //  Datos Pool
    int ancho_filtro_pool = pool[0];
    int saltos_pool = pool[3];
    int ancho_pool = (ancho_convol - ancho_filtro_pool) / saltos_pool + 1;
    
    for (int fil = 0 ; fil < num_filtros; fil++)
    {
        //  Recorre el alto de la matriz de convolucion
        int xp = 0;
        for (int pi = 0; pi < ancho_convol; pi++)
        {
            //  Recorre el alto del filtro
            for (int row = 0; row < ancho_filtro; row++)
            {
                //  Recorre el ancho de la matriz de convolucion - acumula el producto
                for (int pj = 0; pj < ancho_convol; pj++)
                {
                    int xpos_convol = pos_convol + pj + (pi * ancho_convol) + (fil * ancho_convol * ancho_convol);
                    double adelta = capa_cnn[xpos_convol];
                    //  Recorre el filtro y realiza la multiplicacion
                    for (int col = 0; col < ancho_filtro; col++)
                    {
                        
                        int xpos_filtro = pos_filtro + col + (row * ancho_filtro) + (fil*ancho_filtro * ancho_filtro);
                        int xpos_imagen = pos_imagen + col + (pj * saltos) + (row * ancho_imagen) + (pi * ancho_imagen);
                        double afil = capa_cnn[xpos_filtro];
                        double aimg = capa_cnn[xpos_imagen];
//                        cout<<"delta :"<<adelta<<endl;
//                        cout<<"img   :"<<aimg<<endl;
//                        cout<<"Filtro:"<<capa_cnn[xpos_filtro]<<endl<<endl;
                        
                        capa_cnn[xpos_filtro] = capa_cnn[xpos_filtro] - learn_rate * adelta * aimg;
                    }
                }
//                imprimir_filtros(network[0], network[1], pos_cnn[0]);
            }
        }
    }
    
}



void back_pool(int * imagen, int * filtro, int *pool ,int pos_img , int pos_convol , int pos_pool, int pos_delta)
{
    //  Datos imagen
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    int pos_imagen = pos_img; // Posicion inicial de la imagen con padding
    int pos_filtro = pos_imagen + ancho_imagen * ancho_imagen * profundidad_imagen; // Posicion inicial del filtro
    
    //  Datos filtro
    int num_filtros = filtro[4];
    int saltos = filtro[3];
    int ancho_filtro = filtro[0];
    int ancho_convol = (ancho_imagen - ancho_filtro) / saltos + 1;
    
    //  Datos Pool
    int ancho_filtro_pool = pool[0];
    int saltos_pool = pool[3];
    int ancho_pool = (ancho_convol - ancho_filtro_pool) / saltos_pool + 1;
    
    for (int fil = 0 ; fil < num_filtros; fil++)
    {
        //  Recorre el alto de la matriz de convolucion
        int xp = 0;
        for (int pi = 0; pi < ancho_convol; pi++)
        {

            if (pi != 0 && pi % ancho_filtro_pool == 0)
            {
                xp++;
            }
            
            for (int x=0; x < ancho_pool; x++)
            {
                double max = capa_cnn[pos_pool + x + (xp * ancho_pool) + (fil * ancho_pool * ancho_pool)];
                double delta = capa_mlp[1 + pos_delta + x + (xp * ancho_pool) + (fil * ancho_pool * ancho_pool)];
                for (int y = 0; y < ancho_filtro_pool; y++)
                {
                    double valor = capa_cnn[pos_convol + y + (x * ancho_filtro_pool)+(pi * ancho_convol)+(fil * ancho_convol * ancho_convol)];
                    if (valor == max)
                    {
                        capa_cnn[pos_convol + y + (x * ancho_filtro_pool)+(pi * ancho_convol)+(fil * ancho_convol * ancho_convol)] = delta;
                    }else{
                        capa_cnn[pos_convol + y + (x * ancho_filtro_pool)+(pi * ancho_convol)+(fil * ancho_convol * ancho_convol)] = 0;
                    }
                }

            }
        }
    }
    
}

void fully_conected(int pos_pool, int total_datos)
{
    capa_mlp[0] = 1;
    for (int i = 1; i < total_datos + 1; i++) {
        capa_mlp[i] = capa_cnn[pos_pool + i - 1];
    }
    
}

void mlp_fordward()
{

    //  FORWARD PROPAGATION
    for (int c = 0; c < total_capas_mlp-1; c++)
    {
        multiplicacion_punto(pos_mlp[c], pos_mlp[c+1], neurona[c], neurona[c+1]);
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
            point_capa_mlp[ini_matriz + j] = point_capa_mlp[ini_matriz + j] - learn_rate * derivada;
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
        //  En el caso de la primera capa el error solo depende del delta acumulado ya que no hay activacion de la neurona por ser dato de entrada
        for (int i=1; i < cant_input + 1; i++)
        {
            point_capa_mlp[posInput + i] = sumatorias_error[numeroCapa][i];
            //  Inicializamos el acumulador a 0 para proximas iteraciones
            sumatorias_error[numeroCapa][i] = 0;
        }
    
    
    }
}


//  FUNCION DE ACTIVACION
double f_signoid(double numero)
{
    return 1 / (1 + pow(exp(1), -1 * numero));
}



//  FUNCIONES DE AYUDA
void imprimir_imagen(int *imagen, int pos_imagen)
{
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    for (int k =0; k < profundidad_imagen; k++) {
        cout<<"Imagen "<<k<<endl<<endl;
        for (int i=0; i<ancho_imagen; i++) {
            for (int j=0; j<ancho_imagen; j++) {
                cout<<capa_cnn[pos_imagen + j + (i*ancho_imagen) + (k* ancho_imagen*ancho_imagen)]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}

void imprimir_filtros(int *imagen, int *filtro, int pos_imagen)
{
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    int num_filtros = filtro[4];
    int ancho_filtro = filtro[0];
    int pos_filtro = pos_imagen + ancho_imagen * ancho_imagen * profundidad_imagen;
    
    for (int k =0; k < num_filtros; k++)
    {
        cout<<"Filtro "<<k<<endl<<endl;
        for (int i=0; i<ancho_filtro; i++)
        {
            for (int j=0; j<ancho_filtro; j++)
            {
                cout<<capa_cnn[pos_filtro + j + (i*ancho_filtro) + (k * ancho_filtro * ancho_filtro)]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}

void imprimir_convolucion(int *imagen, int *filtro, int pos_convol)
{
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    int num_filtros = filtro[4];
    int saltos = filtro[3];
    int ancho_filtro = filtro[0];
    
    int dim_convol = (ancho_imagen - ancho_filtro) / saltos + 1;
    
    for (int k =0; k < num_filtros; k++) {
        cout<<"Convolucion "<<k<<endl<<endl;
        for (int i=0; i<dim_convol; i++) {
            for (int j=0; j<dim_convol; j++) {
                cout<<capa_cnn[pos_convol + j + (i*dim_convol) + (k* dim_convol*dim_convol)]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}

void imprimir_pool(int *imagen, int *filtro, int *pool, int pos_pool)
{
    int padding = 2 * imagen[3];
    int ancho_imagen = imagen[0] + padding;
    int profundidad_imagen = imagen[2];
    
    int num_filtros = filtro[4];
    int saltos = filtro[3];
    int ancho_filtro = filtro[0];
    
    int dim_convol = (ancho_imagen - ancho_filtro) / saltos + 1;
    
    int ancho_pool = pool[0];
    int saltos_pool = pool[3];
    int dim_pool = (dim_convol - ancho_pool) / saltos_pool + 1;
    
    for (int k =0; k < num_filtros; k++)
    {
        cout<<"Pool "<<k<<endl<<endl;
        for (int i=0; i<dim_pool; i++)
        {
            for (int j=0; j<dim_pool; j++)
            {
                cout<<capa_cnn[pos_pool + j + (i*dim_pool) + (k * dim_pool * dim_pool)]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}

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


//  INICIALIZAMOS LAS ENTRADAS PARA LA PRIMERA CAPA DE CADA CASO DE ENTRENAMIENTO
void copiar_input(_datos *origen, int cantidad)
{
    double aux;
    point_capa_cnn[0] = 1;
    for (int i=0; i<cantidad; i++) {
        aux = origen->data[i];
        point_capa_cnn[i] = aux;
    }
}

//  GENERAMOS UNA ESTRUCTURA CON LA DATA DE ENTRENAMIENTO O TEST
void obtener_data(string archivo,int total_input, int * dim_imagen, double valor_normalizador)
{
    string numero;
    int a,b;
    _datos *nuevo, *puntero_aux;
    
    nuevo = (struct _datos *) malloc (sizeof(struct _datos));
    puntero_aux = nuevo;
    point_data = nuevo;
    
    //  Dimensiones de la Imagen
    int ancho_imagen = dim_imagen[0];
    int pad = dim_imagen[3];
    int total_zeros = ancho_imagen + 2 * pad;
    int padding = 0;
    
    ifstream file(archivo);
    string value;
    while (file.good())
    {
        getline ( file, value, '\n' );
        
        //  declaramos un array
        double *d_array;
        d_array = (double *)malloc(sizeof(double)*total_input);
        
        
        // Primera fila de ceros
        for (int i = 0; i < total_zeros; i++)
        {
            d_array[i] = padding;
        }
        
        int cen1 = 1;
        int cen2 = 0;
        int aux = 1;
        int c = 0;
        //  recorremos cada linea del file e insertamos ceros al inicio y fin
        for (int i = total_zeros; i< total_input - total_zeros; i++)
        {
            if (cen1 == 1 && (aux % ancho_imagen) == 1){
                d_array[i] = padding;
                cen1 = 0;
            }else if (cen2 == 1){
                d_array[i] = padding;
                cen2 = 0;
                aux++;
            }else{
                
                if ((aux % ancho_imagen) == 0)
                {
                    cen1 = 1;
                    cen2 = 1;
                }else{
                    aux++;
                }
                
                a = (int)value.find(';',c);
                b = (int)value.find(';',a+1);
                c = b;
                numero = value.substr(a+1,b-a-1);
                d_array[i] = stof(numero)/valor_normalizador;
//                d_array[i] = stof(numero);
            }
        }
        
        //  Ultima fila de zeros
        for (int i = total_input - total_zeros; i < total_input; i++)
        {
            d_array[i] = padding;
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


//  RESULTADOS DEL TEST Y PRESICION
void detalles_test(int contadorCasos, int contador, int clasificados)
{
    int acurracy = contador * 100 / contadorCasos;
    int noClasificados = contadorCasos - clasificados;
    cout<<endl<<"--------------------------------"<<endl;
    cout<<"Tiempo de Entrenamiento: "<< duration <<endl;
    cout<<"Epocas: "<<total_epocas<<endl;
    cout<<"Learn Rate: "<<learn_rate<<endl;
    cout<<"--------------------------------"<<endl;
    cout<<"Total de Casos: "<<contadorCasos<<endl;
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









#ifndef __CNN__
#define __CNN__

#include        <chrono>
#include "layer.h"
#include <iostream>
#include <random>
#include <fstream>
#include <ctime>

#include <iostream>
#include <random>
using namespace std;

struct _datos
{
    double * data = NULL;
    int target;
};


class _CNN{
    
public:
    int _total_dataset_train;
    int _total_dataset_test;
    int total_array;
    int total_array_aux;
    int dim_image;
    int dimPad_image;
    int channel_image;
    int pad_image;
    int _epoch;
    double _lrate_cnn;
    double _lrate_mlp;
    _datos *dataset;
    double *matrix;
    
    clock_t start;
    double duration;

private:
    
    double *matrix_aux;
    _layer *layer;
    _layer *layer_aux;
    tLayer typeAux;
    int count_layer;
    // Imagen entrada
    int dim_input;
    int deep_input;
    int pad_input;
    int pos_input_start;
    int pos_input_end;
    //  Codificacion de Targets
    double **target;
    int mConf[10][10];

    
public:
    _CNN()
    {
        _epoch = 10;
        _lrate_cnn = 0.1;
        _lrate_mlp = 0.1;
        count_layer = 0;
        dataset = NULL;
        matrix = NULL;
        matrix_aux = NULL;
        typeAux = image;
        
        int dim_target = 10;
        target = (double**)malloc(sizeof(double) * dim_target);
        for (int i = 0; i < dim_target; i++)
        {
            target[i] = (double*)malloc(sizeof(double) * dim_target);
            for (int j = 0; j < dim_target; j++)
            {
                if (i == j)
                    target[i][j] = 0.99;
                else
                    target[i][j] = 0.01;
                
                mConf[i][j] = 0;
            }
        }
    }
    
    ~_CNN()
    {
        delete matrix;
        delete matrix_aux;
        delete layer;
        delete layer_aux;
        delete dataset;
        for (int i = 0; i < 10; i++)
            delete target[i];
        delete target;
    }
    
    void train()
    {
        start = clock();
        cout<<"Train..."<<endl;
        // Recorremos las epocas de entrenamiento
        for (int epo = 0; epo < _epoch; epo++)
        {
//            cout<<"Epoca ["<<epo<<"]"<<endl;
            
            // Recorremos el dataset
            for (int d = 0; d < _total_dataset_train ; d++)
            {
//                cout<<"Data "<<d<<endl;
                copy_image_in_matrix(d);
//                print_image(0,1);
//                print_image(0,2);
//                print_image(0,3);
                // Forward Propagation
                for (int index = 0; index < count_layer; index++)
                {
                    if (layer[index].type == convol)
                    {
                        layer[index].process_convol2();
//                        print_image(index+1,1);
//                        print_image(index+1,2);
//                        print_image(index+1,3);
//                        print_filter_convol(index + 1, 1, 1);
//                        print_filter_convol(index + 1, 1, 2);
//                        print_filter_convol(index + 1, 1, 3);
                    }
                    else if (layer[index].type == pool)
                    {
                        layer[index].process_pool();
//                        print_image(index+1,1);
//                        print_image(index+1,2);
//                        print_image(index+1,3);
                    }
                    else{
                        layer[index].process_mlp();
//                        print_output_mlp(index+1);
                    }
                    
                }
//                cout<<endl;
                
//                print_output_mlp(count_layer);
                
//                print_matrix_total();
                
                // Backward Propagaton
                for (int index = count_layer -1; index > -1; index--)
                {
                    
                    if (index == count_layer -1)
                    {
                        layer[index].process_mlp_back_init(target ,dataset[d].target, layer[index-1].pos_delta_start);
//                        print_output_mlp(index+1);
                    }
                    else if (layer[index].type == fullyC)
                    {
                        layer[index].process_mlp_back(layer[index-1].pos_delta_start);
//                        print_output_mlp(index+1);
                    }
//
                    else if (layer[index].type == pool)
                    {
                        layer[index].process_pool_back(layer[index-1].pos_delta_start);
//                        print_image(index+1, 1,"deltas");
//                        print_image(index+1, 2,"deltas");
//                        print_image(index+1, 3,"deltas");
                    }
                    
                    else if (layer[index].type == convol && layer[index].typeBefore == image)
                    {
                        layer[index].process_convol_back_end();
//                        print_image(index+1, 1,"deltas");
//                        print_image(index+1, 2,"deltas");
//                        print_image(index+1, 3,"deltas");

                    }
                    else if (layer[index].type == convol)
                    {

                        layer[index].process_convol_back(layer[index-1].pos_delta_start);

                    }
                }
            }
        }
        
        cout<<"End Train... "<<endl;
        duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    }
    
    void test()
    {
        cout<<"Test..."<<endl;
        
        int contador_aciertos = 0;
        int clasificados = 0;
        // Recorremos el dataset
        for (int d = 0; d < _total_dataset_test ; d++)
        {
            copy_image_in_matrix(d);
            for (int index = 0; index < count_layer; index++)
            {
                if (layer[index].type == convol)
                {
                    layer[index].process_convol2();
//                    print_image(index+1,1);
                }
                else if (layer[index].type == pool)
                {
                    layer[index].process_pool();
//                    print_image(index+1,1);
                }
                else
                    layer[index].process_mlp();
            }
            
            
            int pos_final = layer[count_layer-1].pos_out_start;
            int total_salidas = layer[count_layer-1].dim_out;
            int res = dataset[d].target;
            double max_final = matrix[pos_final];
            int indice = 0;
            
            for (int i = 1; i < total_salidas; i++)
            {
                if(matrix[pos_final + i] > max_final)
                {
                    max_final = matrix[pos_final + i];
                    indice = i;
                }
            }
            

            if(res == indice)
                contador_aciertos++;
            
            mConf[indice][res] = mConf[indice][res] + 1;
            clasificados++;
        }
        
        detalles_test(contador_aciertos, clasificados);
        matriz_confusion();
    }
    
    
    //  RESULTADOS DEL TEST Y PRESICION
    void detalles_test(int contador, int clasificados)
    {
        double acurracy = double(contador) * 100.00 / double(_total_dataset_test);
        
        int noClasificados = _total_dataset_test - clasificados;
        cout<<endl<<"--------------------------------"<<endl;
        cout<<"Tiempo de Entrenamiento: "<< duration <<endl;
        cout<<"Epocas: "<<_epoch<<endl;
        cout<<"Casos Entrenamiento: "<<_total_dataset_train<<endl;
        cout<<"Learn Rate Cnn: "<<_lrate_cnn<<endl;
        cout<<"Learn Rate Mlp: "<<_lrate_mlp<<endl;
        cout<<"--------------------------------"<<endl;
        cout<<"Casos de Test: "<<_total_dataset_test<<endl;
        cout<<"Aciertos: "<<contador<<endl;
        cout<<"Sin Clasificar: "<<noClasificados<<endl;
        cout<<"Desaciertos: "<<_total_dataset_test - contador - noClasificados<<endl;
        cout<<"Accuracy: "<<acurracy<<"%"<<endl;
    }
    
    //  MATRIZ DE CONFUSION
    void matriz_confusion()
    {
        cout<<endl;
        cout<<"MATRIZ DE CONFUSION"<<endl<<endl;
        for (int i = 0; i < 10; i++)
        {
            int sum = 0;
            cout<<"     "<<i<<" | ";
            for (int j = 0; j < 10; j++) {
                cout<<mConf[i][j]<<"  ";
                sum = sum + mConf[i][j];
            }
            cout<<" : "<<sum<<endl;
        }
        cout<<endl;
    }
    
    void copy_image_in_matrix(int index)
    {
        int total_bucle = dimPad_image * dimPad_image * channel_image;
        for (int i = 0; i < total_bucle; i++)
        {
            double val = dataset[index].data[i];
            matrix[i] = val;
        }
    }
    
    
    void print_image(int indexLayer, int num_slide, string type = "" )
    {
        double * Maux = NULL;
        string strname = "";
        int pos_image = 0;
        int dimension = 0;
        int slide_image;
        if (indexLayer == 0)
        {
            // Para imprimir la imagen de entrada
            Maux = matrix;
            dimension = dimPad_image;
            pos_image = 0 + (num_slide - 1) * dimension * dimension;
            slide_image = channel_image;
            
        }else if(type == "deltas"){
            Maux = matrix_aux;
            dimension = layer[indexLayer-1].dim_out;
            pos_image = layer[indexLayer-1].pos_delta_start + dimension * dimension * (num_slide - 1);
            slide_image = layer[indexLayer-1].deep_out;
            strname = " deltas";
            
        }else{
            // Para imprimir el slide de una convolucion o pool
            Maux = matrix;
            dimension = layer[indexLayer-1].dim_out;
            pos_image = layer[indexLayer-1].pos_out_start + dimension * dimension * (num_slide - 1);
            slide_image = layer[indexLayer-1].deep_out;
        }

        if (num_slide > slide_image)
        {
            cout<<"El slide deseado no existe, debe seleccionar un slide menor a "<<to_string(slide_image)<<endl;
            exit(1);
        }
        
        if(indexLayer == 0)
            cout<<endl<<"Matrix Image [dimension "<<dimension<<"x"<<dimension<<" Slide "<<num_slide<<"]"<<endl<<endl;
        
        else if (layer[indexLayer-1].type == convol)
            cout<<endl<<"Matrix"<<strname<<" Convol [capa "<<indexLayer<<" Slide "<<num_slide<<"]"<<endl<<endl;
        
        else if (layer[indexLayer-1].type == pool)
            cout<<endl<<"Matrix"<<strname<<" Poll [capa "<<indexLayer<<" Slide "<<num_slide<<"]"<<endl<<endl;
        
        
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                cout<<Maux[pos_image + i * dimension + j]<<" ";
            }
            cout<<endl;
        }
    
    }
    
    void print_filter_convol(int indexLayer, int num_filtro, int num_slide)
    {
        int idx = indexLayer - 1;
        int dimension = layer[idx].dim_filter;
        int deep = layer[idx].deep_filter;
        int pos_filter = layer[idx].pos_filter_start + (num_filtro-1) * dimension * dimension * deep + (num_slide-1) * dimension * dimension;
        
        cout<<endl<<"Filtro Convol [capa "<<indexLayer<<" DeepSlide "<<num_slide<<" Slide "<<num_slide<<"]"<<endl<<endl;
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                cout<<pos_filter + (i * dimension) + j<<" : "<<matrix[pos_filter + (i * dimension) + j]<<"|";
//                cout<<matrix[pos_filter + (i * dimension) + j]<<" ";
            }
            cout<<endl;
        }
    
    }
    
    void print_output_mlp(int index_layer_fullyC)
    {
        int i = index_layer_fullyC - 1;
        
        cout<<endl<<"MLP Out [capa "<<index_layer_fullyC<<"]"<<endl<<endl;
        for (int j = 0; j < layer[i].dim_out; j++)
        {
            cout<<matrix[layer[i].pos_out_start + j]<<" ";
        }
        cout<<endl;
    }
    
    void print_input_mlp(int index_layer_fullyC)
    {
        int i = index_layer_fullyC - 1;
        
        cout<<endl<<"MLP Inn [capa "<<index_layer_fullyC<<"]"<<endl<<endl;
        for (int j = 0; j < layer[i].dim_input; j++)
        {
            cout<<matrix[layer[i].pos_input_start + j]<<" ";
        }
        cout<<endl;
    }
    
    void print_weight_mlp(int index_layer_fullyC)
    {
        int i = index_layer_fullyC - 1;
        
        cout<<endl<<"MLP Pesos [Capa  "<<index_layer_fullyC<<"]"<<endl<<endl;
        for (int x = 0; x < layer[i].dim_out; x++)
        {
            for (int y = 0; y < layer[i].dim_input + 1; y++)
            {
                cout<<matrix[layer[i].pos_filter_start + x * (layer[i].dim_input + 1) + y]<<" ";
            }
            cout<<endl;
        }
    }
    
    void set_config(int epoch, double learn_rate_cnn, double learn_rate_mlp)
    {
        _epoch = epoch;
        _lrate_cnn = learn_rate_cnn;
        _lrate_mlp = learn_rate_mlp;
    }

//  VALORES DE LA IMAGEN ORIGINLA DE ENTRADA
    void set_image(int dim_i, int channel_i, int pad_i)
    {
        dim_image    = dim_i;
        dimPad_image = dim_i + 2 * pad_i;
        channel_image= channel_i;
        pad_image    = pad_i;
        dim_input    = dim_i;
        deep_input   = channel_i;
        pad_input    = pad_i;
    }

//  INSERTA CAPA CONVOLUCION O POOL, LOS CANTIDAD DE PARAMETROS VARIAN DEPENDIENDO DEL TIPO DE CAPA O LAYER
    void insert_layer(tLayer type, int val1 = 0, int val2 = 0,int val3 = 0)
    {
        int new_deep = 0;
        double new_dim = 0;
        
        if(layer == NULL)
        {
            // Primera capa insertada
            count_layer++;
            layer = (_layer *)malloc(sizeof(_layer) * count_layer);
        }else
        {
            // Capas insertadas
            count_layer++;
            layer_aux = (_layer *)realloc(layer, sizeof(_layer) * count_layer);

            if (layer_aux != NULL) {
                layer = layer_aux;
                layer_aux = NULL;
            }else{
                delete []layer;
                cout<<"Error allocating memory!"<<endl;
                exit(1);
            }
        }
        
        // Indice de la capa
        int indice = count_layer - 1;
        
        // Input de la capa
        layer[indice].type = type;
        layer[indice].typeBefore = typeAux;
        layer[indice].dim_input = dim_input + pad_input * 2;
        layer[indice].deep_input = deep_input;
        
        
        if (type == convol)
        {
            // Filtros convolucion
            layer[indice].dim_filter = val1;
            layer[indice].num_filter = val2;
            layer[indice].jump = val3;
            layer[indice].deep_filter= deep_input;
            
        }else if (type == pool)
        {
            // Pooling
            layer[indice].dim_filter = val1;
            layer[indice].jump = val2;
            layer[indice].num_filter = deep_input;
            layer[indice].deep_filter = 0;
            
        }else if (type == fullyC)
        {
            // Fully Conected
            if (typeAux != fullyC)
                layer[indice].dim_input = dim_input * dim_input * deep_input;
            else
                layer[indice].dim_input = dim_input;
            
            layer[indice].deep_input = 1;
            layer[indice].dim_out = val1;
            layer[indice].deep_out = 1;
            layer[indice].num_filter = 1;
            new_dim  = val1;
            new_deep = 1;
            
        }else{
            cout<<"Tipo de capa no reconocido"<<endl;
            exit(1);
        }
        
        if (type != fullyC)
        {
            // Hallamos la dimension del resultado
            double dim_input_aux = layer[indice].dim_input;
            double dim_filter_aux= layer[indice].dim_filter;
            double jump_aux      = layer[indice].jump;
            new_deep = layer[indice].num_filter;
            new_dim  = ((dim_input_aux - dim_filter_aux)/jump_aux)+1;
            
            if ((new_dim - (int)new_dim) > 0) {
                cout<<"Error de dimension de filtros!"<<endl;
                exit(1);
            }
            
            // Out o salida de la capa
            layer[indice].dim_out = (int)new_dim;
            layer[indice].deep_out= new_deep;
        }
    
        // Actualizamos la dimension de la matriz resultante para la siguiente capa
        update_input_image(new_dim, new_deep, type);
    }
    
//  GENERAMOS LA ESTRUCTURA Y ASIGNAMOS A CADA CAPA LA POSICION DE INICIO DE SEGMENTOS DEL ARRAY
    void struct_generate()
    {
        cout<<"Generando estructura..."<<endl;
        int total = 0;
        int total_aux = 0;
        for (int i = 0; i < count_layer; i++)
        {
            if (layer[i].type !=  fullyC)
            {
                layer[i].pos_input_start = total;
                
                total = total + layer[i].dim_input * layer[i].dim_input * layer[i].deep_input;
                layer[i].pos_filter_start= total;
                
                total = total + layer[i].dim_filter * layer[i].dim_filter * layer[i].deep_filter * layer[i].num_filter;
                layer[i].pos_out_start   = total;
                
                // Matriz auxiliar
                layer[i].pos_delta_start = total_aux;
                total_aux = total_aux + layer[i].dim_out * layer[i].dim_out * layer[i].deep_out;
            
            }else{
                
                layer[i].pos_input_start = total;
                
                total = total + layer[i].dim_input;
                layer[i].pos_filter_start= total;
                
                total = total + (layer[i].dim_input + 1) * layer[i].dim_out;
                layer[i].pos_out_start   = total;
                
                // Matriz auxiliar
                layer[i].pos_delta_start = total_aux;
                total_aux = total_aux + layer[i].dim_out;
            }
        }
        
        if (layer[count_layer-1].type !=  fullyC)
            total = total + layer[count_layer-1].dim_out * layer[count_layer-1].dim_out * layer[count_layer-1].deep_out;
        else
            total = total + layer[count_layer-1].dim_out;
        
        // Reservamos espacio de memoria para la estructura principal
        matrix = (double *)malloc(sizeof(double) * total);
        matrix_aux = (double *)malloc(sizeof(double) * total_aux);
        
        total_array = total;
        total_array_aux = total_aux;
    }
    
//  INICIALIZAMOS LOS VALORES Y FILTROS DEL CNN
    void struct_initialize()
    {
        cout<<"Inicializando estructura..."<<endl;
        for (int i = 0; i < count_layer; i++)
        {
            // Inicializacion de filtros
            if(layer[i].type ==  convol)
            {
                set_filter(i);
                set_matrix_zeros(i);
            }else if (layer[i].type ==  pool)
            {
                set_pool_matrix_min(i);
            }else if (layer[i].type ==  fullyC)
            {
                set_filter(i);
                set_matrix_zeros(i);
            }
            
            layer[i].matrixL = matrix;
            layer[i].matrixLaux = matrix_aux;
            layer[i].Lrate_cnn = _lrate_cnn;
            layer[i].Lrate_mlp = _lrate_mlp;
        }
        
        for (int i = 0; i < total_array_aux; i++) {
            matrix_aux[i] = 0;
        }
    }
    
//  GENERAMOS LOS FILTROS DE LA CAPA SELECCIONADA
    void set_filter(int index_layer)
    {
        int i = index_layer;
        int total_filter;
        
        if (layer[i].type !=  fullyC)
            total_filter = layer[i].dim_filter * layer[i].dim_filter * layer[i].deep_filter;
        else
            total_filter = (layer[i].dim_input + 1) * layer[i].dim_out;
        
        for (int j = 0; j < layer[i].num_filter; j++)
        {
            int aux = layer[i].pos_filter_start + total_filter * j;
            rand_distribucion_normal(aux, total_filter);
        }
        
    }
    
//  Llenamos la matriz de convol de ceros
    void set_matrix_zeros(int index_layer)
    {
        int i = index_layer;
        int total_convol;
        
        if (layer[i].type !=  fullyC)
            total_convol = layer[i].dim_out * layer[i].dim_out * layer[i].deep_out;
        else
            total_convol = layer[i].dim_out;
        
        for (int j = 0; j < total_convol; j++){
            matrix[layer[i].pos_out_start + j] = 0;
        }
    }
    
//  Llenamos la matriz de pool de un numero muy negativo
    void set_pool_matrix_min(int index)
    {
        int i = index;
        int total_pool = layer[i].dim_out * layer[i].dim_out * layer[i].deep_out;
        for (int j = 0; j < total_pool; j++) {
            matrix[layer[i].pos_out_start + j] = -1000;
        }
    }
    
//  CARGAR DATA DE ARCHIVO EXTERNO
    void load_dataset(tData typeData,string root, int total_data, int value_normalization)
    {
        cout<<"Load Dataset..."<<endl;
        ifstream file(root);
        string cadena;
        string value;
        int index_data;
        dataset = (_datos *)malloc(sizeof(_datos) * total_data);
        
        int total_zeros;
        int padding = 0;
        int counter = 0;
        int total_input = dimPad_image * dimPad_image * channel_image;
        int total_imagen_data;
        
        if (pad_image > 0){
            total_zeros = dimPad_image;
            total_imagen_data = dimPad_image * dimPad_image * channel_image - 2 * pad_image * total_zeros;
        }
        else{
            total_zeros = 0;
            total_imagen_data = dim_image * dim_image * channel_image;
        }
        
        int counter_data = 0;
        while (file.good() && counter < total_data)
        {
            counter_data++;
            int cen1 = 1;
            int cen2 = 0;
            int aux = 1;
            int a = 0;
            int b = 0;
            int c = 0;
            
            index_data = total_zeros * pad_image;
            
            getline(file, cadena, '\n' );
            dataset[counter].data = (double *)malloc(sizeof(double)*total_input);
            
            // Primera fila de ceros
            for (int p = 0; p < pad_image; p++)
                for (int i = 0; i < total_zeros; i++)
                    dataset[counter].data[i] = padding;

            
            //  recorremos cada linea del file e insertamos ceros al inicio y fin
            for (int i = 0; i< total_imagen_data; i++)
            {
                if (cen1 == 1 && (aux % dim_image) == 1 && pad_image > 0){
                    for (int p = 0; p < pad_image; p++)
                        dataset[counter].data[index_data + i + p] = padding;
                    
                    i = i + pad_image - 1;
                    cen1 = 0;
                }else if (cen2 == 1 && pad_image > 0){
                    for (int p = 0; p < pad_image; p++)
                        dataset[counter].data[index_data + i + p] = padding;
                    
                    i = i + pad_image - 1;
                    cen2 = 0;
                    aux++;
                }else{
                    if ((aux % dim_image) == 0)
                    {
                        cen1 = 1;
                        cen2 = 1;
                    }else
                        aux++;

                    a = (int)cadena.find(';',c);
                    b = (int)cadena.find(';',a+1);
                    c = b;
                    value = cadena.substr(a+1,b-a-1);
                    dataset[counter].data[index_data + i] = stof(value)/value_normalization;
                }
            }
            
            index_data = index_data + total_imagen_data;
            
            //  Ultima fila de zeros
            for (int p = 0; p < pad_image; p++)
                for (int i = 0; i < total_zeros; i++)
                    dataset[counter].data[index_data + i] = padding;
            
            //  almacenamos los datos en la lista y enlazamos
            a = (int)cadena.find(';');
            value = cadena.substr(0,a);
            dataset[counter].target = stoi(value);
            counter++;
        }
        
        
        if (counter < total_data)
        {
            cout<<endl<<"Error : El dataset solo contiene "<<counter<<" casos, ingresar una cantidad < o ="<<endl;
            exit(1);
        }
        
        
        if (typeData == tData::train)
            _total_dataset_train = counter_data;
        else
            _total_dataset_test = counter_data;
        
        file.close();
    }
    
//  IMPRIMIMOS EL ARRAY COMPLETO
    void print_matrix_total()
    {
        cout<<endl<<"Array Total :"<<endl<<endl;
        for (int i = 0; i < total_array; i++) {
            cout<<matrix[i]<<" ";
        }
        cout<<endl;
        
        cout<<endl<<"Array Total Aux :"<<endl<<endl;
        for (int i = 0; i < total_array_aux; i++) {
            cout<<matrix_aux[i]<<" ";
        }
        cout<<endl<<endl;
    }
    
    
    void cargar_data_pesos(string archivo)
    {
        cout<<"Load data weight"<<endl;
        string numero;
        int a,b;
        
        int inicio_filtro = dimPad_image * dimPad_image * channel_image;
        int total_filtro_cnn = layer[0].dim_filter * layer[0].dim_filter * layer[0].deep_filter;
        int cantidad_filtros_cnn = layer[0].num_filter;
        
        int inicio_filtro_capa1 = layer[2].pos_filter_start;
        int total_filtro_mlp_capa1 = (layer[2].dim_input + 1) * layer[2].dim_out;
        int inicio_filtro_capa2 = layer[3].pos_filter_start;
        int total_filtro_mlp_capa2 = (layer[3].dim_input + 1) * layer[3].dim_out;
        int lineaCont = 0;
        
        ifstream file(archivo);
        string value;
        while (file.good())
        {
            getline ( file, value, '\n' );
            int c = 0;
            
            if(lineaCont < cantidad_filtros_cnn)
            {
                //  recorremos cada linea del file e insertamos ceros al inicio y fin
                a = (int)value.find(' ',c);
                numero = value.substr(0,a);
                matrix[inicio_filtro + lineaCont * total_filtro_cnn] = stof(numero);
                for (int i = 1; i< total_filtro_cnn; i++)
                {
                    a = (int)value.find(' ',c);
                    b = (int)value.find(' ',a+1);
                    c = b;
                    numero = value.substr(a+1,b-a-1);
                    matrix[inicio_filtro + i + lineaCont * total_filtro_cnn] = stof(numero);
                }
                
            }else if(lineaCont == cantidad_filtros_cnn){
                //  recorremos cada linea del file e insertamos ceros al inicio y fin
                a = (int)value.find(' ',c);
                numero = value.substr(0,a);
                matrix[inicio_filtro_capa1] = stof(numero);
                for (int i = 1; i< total_filtro_mlp_capa1; i++)
                {
                    a = (int)value.find(' ',c);
                    b = (int)value.find(' ',a+1);
                    c = b;
                    numero = value.substr(a+1,b-a-1);
                    matrix[inicio_filtro_capa1 + i] = stof(numero);
                }
            }else if(lineaCont == cantidad_filtros_cnn+1){
                //  recorremos cada linea del file e insertamos ceros al inicio y fin
                a = (int)value.find(' ',c);
                numero = value.substr(0,a);
                matrix[inicio_filtro_capa2] = stof(numero);
                for (int i = 1; i< total_filtro_mlp_capa2; i++)
                {
                    a = (int)value.find(' ',c);
                    b = (int)value.find(' ',a+1);
                    c = b;
                    numero = value.substr(a+1,b-a-1);
                    matrix[inicio_filtro_capa2 + i] = stof(numero);
                }
            }
            
            lineaCont++;
        }
        
    }
    
    void save_train(string root)
    {
        cout<<"Salvando configuracion y pesos..."<<endl;
        
        ofstream osave(root);
        string word = "";
        int stype;
        int sdimension = dim_image;
        int schannel = channel_image;
        int spad = pad_image;
        
//        int spadding = pad_image;
        word = word + "image:"+to_string((int)( image))+" "+to_string(sdimension)+" "+to_string(schannel)+" "+to_string(spad);
        osave<<word<<endl;
        
        //  Guardamos la arquitectura
        for (int i = 0; i < count_layer; i++)
        {
            stype = (int)layer[i].type;
            switch (layer[i].type) {
                case  convol:
                {
                    int sdim_filter = layer[i].dim_filter;
                    int snum_filter = layer[i].num_filter;
                    int sjump = layer[i].jump;
                    word = "convol:"+to_string(stype)+" "+to_string(sdim_filter)+" "+to_string(snum_filter)+" "+to_string(sjump);
                    break;
                }
                case  pool:
                {
                    int sdim_filter = layer[i].dim_filter;
                    int sjump = layer[i].jump;
                    word = "pool:"+to_string(stype)+" "+to_string(sdim_filter)+" "+to_string(sjump);
                    break;
                }  
                case  fullyC:
                {
                    int sdim_out = layer[i].dim_out;
                    word = "fullyC:"+to_string(stype)+" "+to_string(sdim_out);
                    break;
                }
                default:
                    break;
            }
            osave<<word<<endl;
        }
        
        //  Guardamos los pesos
        int spos_filter = 0;
        int stotal_filter = 0;
        int sindex = 0;
        string stype_Layer;
        for (int i = 0; i < count_layer; i++)
        {
            stype_Layer = to_string((int)layer[i].type);
            sindex = i;
            switch (layer[i].type)
            {
                case  convol:
                {
                    spos_filter = layer[i].pos_filter_start;
                    stotal_filter = layer[i].dim_filter * layer[i].dim_filter * layer[i].deep_filter * layer[i].num_filter;
                    
                    osave<<endl;
                    break;
                }
                case  fullyC:
                {
                    spos_filter = layer[i].pos_filter_start;
                    stotal_filter = (layer[i].dim_input+1) * layer[i].dim_out;
                    osave<<endl;
                    break;
                }
                default:
                {
                    spos_filter = 0;
                    stotal_filter = 0;
                    break;
                }
            }
            
            if (spos_filter > 0)
            {
                word = stype_Layer+":"+to_string(sindex)+":"+to_string(matrix[spos_filter]);
                for (int i = 1; i < stotal_filter ; i++)
                {
                    string val2 = to_string(matrix[spos_filter + i]);
                    word = word +" "+ val2 ;
                }
                osave<<word;
            }
        }
    
    }

    void load_train(string root)
    {
        int cen = 0;
        ifstream file(root);
        string value;
        int lineCont = 0;
        int type,a,b;
        layer = NULL;
        count_layer = 0;
        cout<<"Cargando configuracion de la cnn..."<<endl;
        while (file.good())
        {
            getline ( file, value, '\n' );

            if (cen == 0 && value != "")
            {
                a = (int)value.find(':');
                b = (int)value.find(' ');
                type = stoi(value.substr(a+1,b-a));
                //  Cargamos la configuracion de la red
                switch (type) {
                    case (int) image:
                    {
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int ldim_image = stoi(value.substr(a,b-a));
                        
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int lchannel = stoi(value.substr(a,b-a));
                        
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int lpad_image = stoi(value.substr(a,b-a));
                        
                        set_image(ldim_image, lchannel, lpad_image);
                        cout<<"    Set Image: "<<ldim_image<<" "<<lchannel<<" "<<lpad_image<<" "<<endl;
                        break;
                    }
                    case (int) convol:
                    {
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int ldim_conv = stoi(value.substr(a,b-a));
                        
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int lnum_filter = stoi(value.substr(a,b-a));
                        
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int ljump_conv = stoi(value.substr(a,b-a));
                        
                        insert_layer( convol,ldim_conv,lnum_filter,ljump_conv);
                        cout<<"    Insert Layer Convol: "<<ldim_conv<<" "<<lnum_filter<<" "<<ljump_conv<<" "<<endl;
                        break;
                    }
                    case (int) pool:
                    {
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int ldim_pool = stoi(value.substr(a,b-a));
                        
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int ljump_pool = stoi(value.substr(a,b-a));
                        
                        insert_layer( pool,ldim_pool,ljump_pool);
                        cout<<"    Insert Layer Pool: "<<ldim_pool<<" "<<ljump_pool<<endl;
                        break;
                    }
                    case (int) fullyC:
                    {
                        a = b+1;
                        b = (int)value.find(" ",a);
                        int ldim_fc = stoi(value.substr(a,b-a));
                        
                        insert_layer( fullyC,ldim_fc);
                        cout<<"    Insert Layer Fc: "<<ldim_fc<<endl;
                        break;
                    }
                        
                    default:
                        break;
                }
            }else if(value == "")
            {
                cen = 1; // Centinela para dejar de insertar capas
                struct_generate(); // Generamos la estructura con las capas cargadas
                struct_initialize(); // Inicializamos la estructura
                cout<<"Cargando los pesos entrenados"<<endl;
            }else
            {
                // Ingresamos los pesos entrenados
                a = (int)value.find(':');
                b = (int)value.find(':',a+1);
                string valux = value.substr(0,a);
                type = stoi(valux);
                valux = value.substr(a+1,b-a-1);
                int index = stoi(valux);
                int lpos_filter = 0;
                int ltotal_filter = 0;
                switch (type)
                {
                    case (int) convol:
                    {
                        lpos_filter = layer[index].pos_filter_start;
                        ltotal_filter = layer[index].dim_filter * layer[index].dim_filter * layer[index].deep_filter * layer[index].num_filter;

                        break;
                    }
                    case (int) fullyC:
                    {
                        lpos_filter = layer[index].pos_filter_start;
                        ltotal_filter = (layer[index].dim_input+1) * layer[index].dim_out;

                        break;
                    }
                    default:
                        break;
                }
                
                for (int i = 0; i < ltotal_filter ; i++)
                {
                    a = b+1;
                    b = (int)value.find(" ",a);
                    valux = value.substr(a,b-a);
                    matrix[lpos_filter + i] = stod(valux);
                }
            }
            
            lineCont++;
        }
        cout<<"Carga Completa!"<<endl;
        cout<<"--------------------------------"<<endl;
    }
    
private :
//  NUMEROS ALEATORIOA DE DISTRIBUCION NORMAL
    void rand_distribucion_normal(int pos_init, int num_random)
    {
//        srand(time(0));
        double seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        double sigma = sqrt(double(2)/num_random);
        normal_distribution<double> distribution(0,sigma);
        for (int i = 0; i < num_random;i++)
        {
            matrix[pos_init + i] = distribution(generator);
//            matrix[pos_init + i] = 0.23;
        };
    }
    
    void update_input_image(int new_dim, int new_deep, tLayer new_type)
    {
        dim_input = new_dim;
        deep_input = new_deep;
        pad_input = 0;
        typeAux = new_type;
    }
};


#endif

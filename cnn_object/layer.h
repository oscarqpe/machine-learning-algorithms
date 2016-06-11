#ifndef __LAYER__
#define __LAYER__

#include <cstddef>
#include <cmath>

using namespace std;



enum  tLayer{image = 0, convol = 1, pool = 2, fullyC = 3};
enum  tData{train, test};
class _layer
{

public:
    
    tLayer type;
    tLayer typeBefore;
    
    double *matrixL;
    double *matrixLaux;
    double Lrate_cnn;
    double Lrate_mlp;
    
    // Imagen entrada
    int dim_input;
    int deep_input;
    int pad_input;
    int pos_input_start;
    
    // Filtros
    int dim_filter;
    int deep_filter;
    int num_filter;
    int jump;
    int pos_filter_start;

    
    // Salida
    int dim_out;
    int deep_out;
    int pos_out_start;
    
    // Convol Deltas
    int pos_delta_start;
//    char * activate;
    
    
public:
    _layer(){
        matrixL = NULL;
        matrixLaux = NULL;
    };
    ~_layer(){
//        delete activate;
//        delete matrixL;
//        delete matrixLaux;
    }
    
    void process_convol()
    {
        for (int indexFil = 0 ; indexFil < num_filter; indexFil++)
        {
            int pos_slide_cube_filter = indexFil * dim_filter * dim_filter * deep_filter;
            
            for (int rowImg = 0; rowImg < dim_out; rowImg++) // Recorre el alto de la matriz de convolucion
            {
                int pos_row_image = rowImg * dim_input * jump;
                
                for (int rowFil = 0; rowFil < dim_filter; rowFil++) // Recorre el alto del filtro
                {
                    int pos_rowFil_image  = rowFil * dim_input;
                    int pos_rowFil_filter = rowFil * dim_filter;
                    
                    for (int colImg = 0; colImg < dim_out; colImg++) // Recorre el ancho de la matriz de convolucion - acumula el producto
                    {
                        int pos_col_image = colImg * jump;
                        double sum_slides = 0;
                        
                        for (int deep = 0; deep < deep_input; deep++)
                        {
                            int slide_image  = deep * dim_input * dim_input;
                            int slide_filter = deep * dim_filter* dim_filter;
                            double sum = 0;
                            
                            for (int colFil = 0; colFil < dim_filter; colFil++) // Recorre el ancho del filtro y realiza la multiplicacion
                            {
                                int xpos_filtro = pos_filter_start + pos_slide_cube_filter + pos_rowFil_filter + slide_filter + colFil;
                                int xpos_image = pos_input_start  + pos_row_image + pos_rowFil_image + pos_col_image + slide_image + colFil ;
                                
                                double afil = matrixL[xpos_filtro];
                                double aimg = matrixL[xpos_image];
                                sum = sum + afil * aimg;
                                
                            }
                            sum_slides = sum_slides + sum;
                        }
                        
                        int xpos_convol = pos_out_start + colImg + (rowImg * dim_out) + (indexFil * dim_out * dim_out);
                        matrixL[xpos_convol] = matrixL[xpos_convol] + sum_slides;
                        
                        
                        // Reiniciamos la matriz auxiliar a ceros
                        matrixLaux[pos_delta_start + (indexFil * dim_out * dim_out) + (rowImg * dim_out) + colImg] = 0;
                    }
                }
            }
        }
    }
    
    void process_convol2()
    {
        for (int indexFil = 0 ; indexFil < num_filter; indexFil++)
        {
            int pos_slide_matrixL_out = indexFil * dim_out * dim_out;
            int pos_numfilter_convol = indexFil * dim_filter * dim_filter * deep_filter;
            // Recorremos la convolucion
            for (int rowConv = 0; rowConv < dim_out; rowConv++)
            {
                int pos_row_input_jump = rowConv * dim_input * jump;
                
                for (int colConv = 0; colConv < dim_out; colConv++)
                {
                    int pos_col_input_jump = colConv * jump;
                    double sum_slides = 0;
                    
                    // Recorremos la profundidad de la imagen o input
                    for (int deep = 0; deep < deep_input; deep++)
                    {
                        int pos_slide_input = deep * dim_input * dim_input;
                        int pos_slide_filter = deep * dim_filter * dim_filter;
                        double sum = 0;
                        
                        // Recorremos el filtro
                        for (int rowFil = 0; rowFil < dim_filter; rowFil++)
                        {
                            int pos_row_input  = rowFil * dim_input;
                            int pos_row_filter  = rowFil * dim_filter;
                            
                            for (int colFil = 0; colFil < dim_filter; colFil++)
                            {
                                double val_input  = matrixL[pos_input_start + pos_row_input_jump + pos_col_input_jump + pos_slide_input + pos_row_input + colFil];
                                double val_filter = matrixL[pos_filter_start + pos_numfilter_convol + pos_slide_filter + pos_row_filter + colFil];

                                sum = sum + val_input * val_filter;
                            }
                        }
                        
                        sum_slides = sum_slides + sum;
                    }
                    
                    // RELU
//                    matrixL[pos_out_start + pos_slide_matrixL_out + (rowConv * dim_out) + colConv] = sum_slides < 0 ? 0 : sum_slides;
                    // SIN RELU
                    matrixL[pos_out_start + pos_slide_matrixL_out + (rowConv * dim_out) + colConv] = sum_slides;

                    
                    // Reiniciamos la matriz auxiliar a ceros
                    matrixLaux[pos_delta_start + pos_slide_matrixL_out + (rowConv * dim_out) + colConv] = 0;
                }
            }
        }
    }
    
//  ACTUALIZAMOS LOS FILTROS Y GENERAMOS DELTAS PARA LA CAPA SIGUIENTE
    void process_convol_back(int pos_delta_before)
    {
        for (int Fil = 0; Fil < num_filter; Fil++)
        {
            //  Numero de Slide
            int pos_slide_convol = Fil * dim_out * dim_out;
            int pos_slide_filter = Fil * dim_filter * dim_filter * deep_filter;
            
            for (int slideIn = 0 ; slideIn < deep_filter; slideIn++)
            {
                // Slide por filtro
                int pos_slide_input = slideIn * dim_input * dim_input;
                int pos_slide_filter_deep = slideIn * dim_filter * dim_filter;
                
                // Recorremos un slide completo multiplicacion punto punto
                for (int rowFil = 0; rowFil < dim_filter; rowFil++)
                {
                    for (int colFil = 0; colFil < dim_filter; colFil++)
                    {
                        double sum = 0;
                        int pos_update_filter = pos_filter_start + pos_slide_filter + pos_slide_filter_deep + (rowFil * dim_filter) + colFil;
                        double val_filter = matrixL[pos_update_filter];

                        // Recorremos la matriz de convolucion
                        for (int rowConv = 0; rowConv < dim_out; rowConv++)
                        {
                            int pos_row_input_jump = rowConv * dim_input * jump;
                            
                            for (int colConv = 0; colConv < dim_out; colConv++)
                            {
                                int pos_col_input_jump = colConv * jump;
                                double val_convol_delta = matrixLaux[pos_delta_start + pos_slide_convol + (rowConv * dim_out) + colConv];
                                double val_input  = matrixL[pos_input_start + pos_slide_input + (rowFil * dim_input) + colFil + pos_row_input_jump + pos_col_input_jump];
                                
                                // Acumulamos los deltas de la siguiente capa, en la estructura axiliar
                                int pos_new_delta = pos_delta_before + pos_slide_input + (rowFil * dim_input) + colFil + pos_row_input_jump + pos_col_input_jump;
                                matrixLaux[pos_new_delta] = matrixLaux[pos_new_delta] + val_filter * val_convol_delta;
                                
                                // Acumulamos el error total de la convolucion para actualizar los filtros
                                sum = sum + val_convol_delta * val_input;
                            }
                        }
                        matrixL[pos_update_filter] = matrixL[pos_update_filter] - Lrate_cnn * sum;
                    }
                }
            }
        }
    }
    
//  ACTUALIZAMOS LOS FILTROS NO GENERAMOS DELTAS YA QUE ES LA CAPA FINAL
    void process_convol_back_end()
    {
        for (int Fil = 0; Fil < num_filter; Fil++)
        {
            //  Numero de Slide
            int pos_slide_convol = Fil * dim_out * dim_out;
            int pos_slide_filter = Fil * dim_filter * dim_filter * deep_filter;
            
            for (int slideIn = 0 ; slideIn < deep_filter; slideIn++)
            {
                // Slide por filtro
                int pos_slide_input = slideIn * dim_input * dim_input;
                int pos_slide_filter_deep = slideIn * dim_filter * dim_filter;
                
                // Recorremos un slide completo multiplicacion punto punto
                for (int rowFil = 0; rowFil < dim_filter; rowFil++)
                {
                    for (int colFil = 0; colFil < dim_filter; colFil++)
                    {
                        double sum = 0;
                        int pos_update_filter = pos_filter_start + pos_slide_filter + pos_slide_filter_deep + (rowFil * dim_filter) + colFil;
                        
                        // Recorremos la matriz de convolucion
                        for (int rowConv = 0; rowConv < dim_out; rowConv++)
                        {
                            int pos_row_input_jump = rowConv * dim_input * jump;
                            
                            for (int colConv = 0; colConv < dim_out; colConv++)
                            {
                                int pos_col_input_jump = colConv * jump;
                                double val_convol_delta = matrixLaux[pos_delta_start + pos_slide_convol + (rowConv * dim_out) + colConv];
                                double val_input  = matrixL[pos_input_start + pos_slide_input + (rowFil * dim_input) + colFil + pos_row_input_jump + pos_col_input_jump];
                                
                                // Acumulamos el error total de la convolucion para actualizar los filtros
                                sum = sum + val_convol_delta * val_input;
                            }
                        }
                        matrixL[pos_update_filter] = matrixL[pos_update_filter] - Lrate_cnn * sum;
                    }
                }
            }
        }
    }

    void process_mlp_back_init(double **target , int indexObj, int pos_delta_before)
    {
        for (int i = 0; i < dim_out; i++)
        {
            double output = matrixL[pos_out_start + i];
            double obj = target[indexObj][i];
            matrixL[pos_out_start + i] = (output - obj) * output * (1 - output);
//            cout<<matrixL[pos_out_start + i]<<" ";
        }
//        cout<<endl;
        process_mlp_back(pos_delta_before);
    }
    
//  ACTUALIZAR PESOS
    void process_mlp_back(int pos_delta_before)
    {
        double derivada;
        double weight;
        double input;
        double delta;
        double sumDelta;
        int pos_weight_update;
        
        //  Recorremos los deltas de la capa que hallamos anteriormente - hay un delta por neurona
        for (int i = 0; i < dim_out; i++)
        {
            // Actualizamos el peso del Bias
            input = 1;
            delta  = matrixL[pos_out_start + i];
            derivada = delta * input;
            matrixL[pos_filter_start] = matrixL[pos_filter_start] - Lrate_mlp * derivada;
            
            //  Recorremos las entradas de las neuronas de la capa siguiente
            for (int j=0; j < dim_input; j++)
            {
                pos_weight_update = pos_filter_start + 1 + (i * (dim_input+1)) + j;
                weight = matrixL[pos_weight_update];
                input = matrixL[pos_input_start + j];
                delta = matrixL[pos_out_start + i];
                
                //  Acumulamos el error cometido para casa neurona j esima de la capa l
                matrixLaux[pos_delta_before + j] = matrixLaux[pos_delta_before + j] + (delta * weight);
                //  La derivada del error total respecto a un peso wi es igual al delta x por el input de la neurona relacionada al peso
                derivada = delta * input;
                //  Actualizamos el peso
                matrixL[pos_weight_update] = matrixL[pos_weight_update] - Lrate_mlp * derivada;
            }
        }
//
        if (typeBefore == fullyC)
        {
            
            for (int i=0; i < dim_input; i++)
            {
                //  Almacenamos el delta para la iteracion de la siguiente capa, en nuestro array esto lo almacenamos en los input de salida de la proxima capa ya que estos no se volveran a utilizar, de esta manera reducimos el tamaño de nuestro array
                input = matrixL[pos_input_start + i];
                sumDelta = matrixLaux[pos_delta_before + i];
                matrixL[pos_input_start + i] = sumDelta  * (input * (1 - input));
                //  Inicializamos el acumulador auxiliar a 0 para proximas iteraciones
                matrixLaux[pos_delta_before + i] = 0;
            }
        }
//        else
//        {
            //  En el caso de la primera capa el error solo depende del delta acumulado ya que no hay activacion de la neurona por ser dato de entrada
//            for (int i = 0; i < dim_input; i++)
//            {
//                sumDelta = matrixLaux[pos_delta_before + i];
////                matrixLaux[pos_input_start + i] = sumDelta;
////                //  Inicializamos el acumulador a 0 para proximas iteraciones
////                matrixLaux[pos_delta_before + i] = 0;
//            }
//        }
    }
    
    void process_pool()
    {
        for (int indexFil = 0 ; indexFil < deep_input; indexFil++)
        {
            int pos_slide_matrixL_imagen = indexFil * dim_input * dim_input;
            
            for (int rowImg = 0; rowImg < dim_out; rowImg++)
            {
                int pos_row_image = rowImg * dim_input * jump;
                
                for (int colImg = 0; colImg < dim_out; colImg++)
                {
                    int pos_col_image = colImg * jump;
                    double max = -1000;

                    for (int rowFil = 0; rowFil < dim_filter; rowFil++)
                    {
                        int pos_rowFil_image  = rowFil * dim_input;
                        
                        for (int colFil = 0; colFil < dim_filter; colFil++)
                        {
                            int xpos_image = pos_input_start + pos_slide_matrixL_imagen + pos_row_image +pos_col_image + pos_rowFil_image + colFil;
                            double value = matrixL[xpos_image];
                            
                            if (value > max)
                            {
                                max = value;
                            }
                        }
                    }
                    
                    int xpos_pool = pos_out_start + (indexFil * dim_out * dim_out) + (rowImg * dim_out) + colImg;
                    matrixL[xpos_pool] = max;
                    
                    // Reiniciamos la matriz auxiliar a ceros
                    matrixLaux[pos_delta_start + (indexFil * dim_out * dim_out) + (rowImg * dim_out) + colImg] = 0;
                }
            }
        }
    }
    
    void process_pool_back(int pos_delta_before = 0)
    {
        for (int islide = 0 ; islide < deep_out; islide++)
        {
            int pos_slide_matrix_pool = islide * dim_out * dim_out;
            int pos_slide_matrix_input = islide * dim_input * dim_input;
            
            for (int pRow = 0; pRow < dim_out ; pRow++)
            {
                int pos_row_pool = pRow * dim_out;
                int pos_row_jump = pRow * dim_input * jump;
                for (int pCol = 0; pCol < dim_out ; pCol++)
                {
                    int pos_col_jump = pCol * jump;
                    double max   = matrixL[pos_out_start + pos_slide_matrix_pool + pos_row_pool + pCol];
                    double delta = matrixLaux[pos_delta_start + pos_slide_matrix_pool + pos_row_pool + pCol];
                    int cen   = 0;
                    for (int filRow = 0; filRow < dim_filter; filRow++)
                    {
                        int pos_row_input = filRow * dim_input;
                        
                        for (int filCol = 0; filCol < dim_filter; filCol++)
                        {
                            int new_pos_value = pos_slide_matrix_input + pos_row_jump + pos_col_jump + pos_row_input + filCol;
                            double value = matrixL[pos_input_start + new_pos_value];
                            int new_pos_delta_aux = pos_delta_before + new_pos_value;
                            
                            if (value == max && cen == 0)
//                            if (value == max)
                            {
                                matrixLaux[new_pos_delta_aux] = matrixLaux[new_pos_delta_aux] + delta;
                                cen = 1;
                            }else
                                matrixLaux[new_pos_delta_aux] = matrixLaux[new_pos_delta_aux] + 0;
                        }
                    }
                }
            }
        }

    }
    
    void process_mlp()
    {
        double weigth = 0;
        double val_in  = 0;
        double sumatoria = 0;
        
        for (int i = 0; i< dim_out; i++)
        {
            int pos_row_fil = (dim_input + 1) * i;
            sumatoria = matrixL[pos_filter_start + pos_row_fil] * 1;
            for (int j = 0; j < dim_input ; j++)
            {
                weigth = matrixL[pos_filter_start + 1 + pos_row_fil + j];
                val_in  = matrixL[pos_input_start + j];
                sumatoria = sumatoria + weigth * val_in;
            }
            matrixL[pos_out_start + i] = f_signoid(sumatoria);
        }
    }
    
private:
    
//  FUNCION DE ACTIVACION
    double f_signoid(double number)
    {
        return 1 / (1 + pow(exp(1), -1 * number));
    }
};


#endif

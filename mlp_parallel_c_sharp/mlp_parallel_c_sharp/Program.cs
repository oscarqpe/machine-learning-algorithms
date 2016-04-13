using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mlp_parallel_c_sharp
{
    class Program
    {
        
        static void Main(string[] args)
        {
            int numDatosEntrada = 2;
            int numDatosSalida = 2;
            int numCapas = 3;
            int neuronByLayer = 2;
    
            double[] real = new double[numDatosSalida + 1];
            double[] error = new double[numDatosSalida + 1];

            //Console.WriteLine("Ingrese el numero de capas intermedias: ");
            //string line = Console.ReadLine();
            ///numCapas = int.Parse(line);
            //line = Console.ReadLine();

            //numCapas += 2;
            int[] numNeuronasPorCapa = new int[numCapas];
            int[] numNeuronasPorCapaAnterior = new int[numCapas + 1];
            double[,,] MLP = new double[numCapas, neuronByLayer + 1, neuronByLayer + 4];

            // input layer
            numNeuronasPorCapa[0] = numDatosEntrada + 1;
            for (int i = 0; i < numNeuronasPorCapa[0]; i++)
            {
                // 3 Filas por neurona
                //MLP[0][i] = (double*)malloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                {
                    MLP[0, i ,j] = 0.0;
                }
            }
            //Capas intermedias
            for (int i = 1; i < numCapas - 1; i++)
            {
                int numNeuronas = neuronByLayer;
                //cout << "Ingrese el numero de neuronas en la capa " << i << ": ";
                //cin >> numNeuronas;
                numNeuronas += 1;
                numNeuronasPorCapa[i] = numNeuronas;
                //Neuronas por capa, ya se le sumo una del bias
                //MLP[i] = (double**)malloc(sizeof(double) * numNeuronas);
                for (int j = 0; j < numNeuronas; j++)
                {
                    //Filas por neurona por capa mas 3 del net, out y delta
                    //MLP[i][j] = (double*)malloc(sizeof(double) * (numNeuronasPorCapa[i - 1] + 3));
                    for (int k = 0; k < numNeuronasPorCapa[i - 1] + 3; k++)
                    {
                        //  cout<<"Num neuronas anterior: "<<numNeuronasPorCapa[i]<<" "<< i << "-"<<j<<"-"<<k<<endl;
                        MLP[i, j, k] = 0;
                    }
                }
            }
            //Capa de salida
            numNeuronasPorCapa[numCapas - 1] = numDatosSalida + 1;
            //MLP[numCapas - 1] = (double**)malloc(sizeof(double) * numNeuronasPorCapa[numCapas - 1]);
            for (int i = 0; i < numNeuronasPorCapa[numCapas - 1]; i++)
            {
                //MLP[numCapas - 1][i] = (double*)malloc(sizeof(double) * (numNeuronasPorCapa[numCapas - 2] + 3));
                for (int j = 0; j < numNeuronasPorCapa[numCapas - 2] + 3; j++)
                {
                    MLP[numCapas - 1, i, j] = 0;
                    // cout<<"Num neuronas anterior: "<<numNeuronasPorCapa[numCapas-1]<<" "<< numCapas-1 << "-"<<i<<"-"<<j<<endl;
                }
            }

            //TRUQUITO JIJI
            numNeuronasPorCapaAnterior[0] = 0;
            for (int i = 1 ; i < numCapas + 1; i++) {
                numNeuronasPorCapaAnterior[i] = numNeuronasPorCapa[i - 1];
            }
    
            int[] numFilasPorCapa = new int[numCapas];
            for (int i = 0 ; i < numCapas; i++) {
                numFilasPorCapa[i] = numNeuronasPorCapaAnterior[i] + 3;
            }
            /////////////////////////////////////////////////
            /////////////  SETEAR MATRIZ    /////////////////
            /////////////////////////////////////////////////

            //    Inicializar bias
            for (int i = 0; i < numCapas; i++)
            {
                MLP[i, 0, numNeuronasPorCapaAnterior[i] + 1] = 1;
            }
            //Setear la matriz de ejemplo
            MLP[0, 1, 1] = 0.05;
            MLP[0, 2, 1] = 0.10;

            MLP[1, 1, 0] = 0.35;//b1
            MLP[1, 2, 0] = 0.35;//b1
            MLP[1, 1, 1] = 0.15;//w1
            MLP[1, 1, 2] = 0.20;//w2
            MLP[1, 2, 1] = 0.25;//w3
            MLP[1, 2, 2] = 0.30;//w4

            MLP[2, 1, 0] = 0.60;//b2
            MLP[2, 2, 0] = 0.60;//b2
            MLP[2, 1, 1] = 0.40;//w5
            MLP[2, 1, 2] = 0.45;//w6
            MLP[2, 2, 1] = 0.50;//w7
            MLP[2, 2, 2] = 0.55;//w8

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

            for (int i = 1; i < numCapas; i++)
            {
                //for (int j = 1; j < numNeuronasPorCapa[i]; j++)
                //{
                Parallel.For(1, numNeuronasPorCapa[i], index_j =>
                {
                    MLP[i, index_j, numFilasPorCapa[i] - 3] = 0;//Resetear NET

                    /*for (int k = 0; k < numFilasPorCapa[i] - 3; k++)
                    {
                        MLP[i, j, numFilasPorCapa[i] - 3] += MLP[i - 1, k, numFilasPorCapa[i - 1] - 2] * MLP[i, j, k];//NET
                    }*/
                    Parallel.For(0, numFilasPorCapa[i] - 3,
                        index_k =>
                        {
                            MLP[i, index_j, numFilasPorCapa[i] - 3] += MLP[i - 1, index_k, numFilasPorCapa[i - 1] - 2] * MLP[i, index_j, index_k];//NET
                        });
                    MLP[i, index_j, numFilasPorCapa[i] - 2] = 1 / (1 + (float)Math.Exp(-MLP[i, index_j, numFilasPorCapa[i] - 3]));//OUT
                });
                //}
            }

            /////////////////////////////////////////////////
            ////////////  IMPRIMIR MATRIZ    ////////////////
            /////////////////////////////////////////////////

            for (int i = 0; i < numCapas; i++)
            {
                for (int j = 0; j < numNeuronasPorCapa[i]; j++)
                {
                    for (int k = 0; k < numNeuronasPorCapaAnterior[i] + 3; k++)
                    {
                        Console.Write(MLP[i, j, k] + "\t");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine("\n");
            }
            Console.WriteLine("Que paso aqui!\n");

            for (int i = 0; i <= numDatosSalida; i++)
            {
                Console.WriteLine(error[i]);
            }

            Console.WriteLine("hola mundo");
            Console.ReadKey();
        }
    }
}

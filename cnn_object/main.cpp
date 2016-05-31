//
//  main.cpp
//  testC
//
//  Created by Ricardo Coronado on 25/05/16.
//  Copyright © 2016 Ricardo Coronado. All rights reserved.
//

#include <iostream>
#include <random>
#include <fstream>
#include <ctime>
#include "layer.h"
#include "cnn.h"

using namespace std;

int main(int argc, const char * argv[])
{
//    srand(time(0));
    _CNN ml;

//  MNIST DE 8X8
//    ml.set_config(300,0.001,0.8); // (Epocas, Lrate CNN, Lrate MLP)
//    ml.set_image(8,1,1); // (Dimension, Channel, padding)
//    ml.insert_layer(tLayer::convol,3,6,1);  // (convol, dim_filter, num_filter, jump)
//    ml.insert_layer(tLayer::pool,2,2);      // (pool, dim_pool, jump)
//    ml.insert_layer(tLayer::fullyC,16);
//    ml.insert_layer(tLayer::fullyC,10);
//
//    ml.struct_generate();
//    ml.struct_initialize();
//    ml.load_train("PruebaMnis8x8.csv");
//    ml.load_dataset(tData::train,"mnist.csv", 947, 17);
//    ml.train();
//    ml.save_train("PruebaMnis8x8.csv");
//    ml.load_dataset(tData::test, "mnist_test.csv", 800, 17);
//    ml.test();


//  MNIST DE 28X28
    ml.set_config(1, 0.001, 0.1); // (Epocas, Lrate CNN, Lrate MLP)
    ml.set_image(28,1,0); // (Dimension, Channel, padding)
    ml.insert_layer(tLayer::convol,5,20,1);  // (convol, dim_filter, num_filter, jump)
    ml.insert_layer(tLayer::pool,2,2);      // (pool, dim_pool, jump)
    ml.insert_layer(tLayer::fullyC,100);
    ml.insert_layer(tLayer::fullyC,10);
    ml.struct_generate();
    ml.struct_initialize();
    ml.load_dataset(tData::train,"mnist_train_28x28.csv", 50000, 1);
    ml.train();
//    ml.save_train("PruebaMnist28X28.csv");
//    ml.load_train("Prueba1.csv");
    ml.load_dataset(tData::test, "mnist_test_28x28.csv", 10000, 1);
    ml.test();

//  MNIST DE 28X28 opcion 2
//    ml.set_config(10, 0.001, 0.08); // (Epocas, Lrate CNN, Lrate MLP)
//    ml.set_image(28,1,0); // (Dimension, Channel, padding)
//    ml.insert_layer(tLayer::convol,5,20,1);  // (convol, dim_filter, num_filter, jump)
//    ml.insert_layer(tLayer::pool,2,2);      // (pool, dim_pool, jump)
//    ml.insert_layer(tLayer::convol,5,40,1);
//    ml.insert_layer(tLayer::pool,2,2);
//    ml.insert_layer(tLayer::fullyC,100);
//    ml.insert_layer(tLayer::fullyC,10);
//    ml.struct_generate();
//    ml.struct_initialize();
//    ml.load_dataset(tData::train,"mnist_train_28x28.csv", 50000, 1);
//    ml.train();
//    ml.load_dataset(tData::test, "mnist_test_28x28.csv", 10000, 1);
//    ml.test();
    

//  CIFAR-10
    
//    ml.set_config(2, 0.001, 0.3); // (Epocas, Lrate CNN, Lrate MLP)
//    ml.set_image(32,3,0); // (Dimension, Channel, padding)
//    ml.insert_layer(tLayer::convol,5,32,1);  // (convol, dim_filter, num_filter, jump)
//    ml.insert_layer(tLayer::pool,2,2);      // (pool, dim_pool, jump)
//    ml.insert_layer(tLayer::convol,5,32,1);
//    ml.insert_layer(tLayer::pool,2,2);
//    ml.insert_layer(tLayer::convol,5,64,1);
//    ml.insert_layer(tLayer::fullyC,64);
//    ml.insert_layer(tLayer::fullyC,10);
//    ml.struct_generate();
//    ml.struct_initialize();
//    ml.load_train("saveCifar.csv");
//    ml.load_dataset(tData::train,"cifarTrain.csv", 50000 , 255);
//    ml.train();
//    ml.load_train("saveCifar2.csv");
//    ml.load_dataset(tData::test, "cifarTest.csv", 10000, 255);
//    ml.test();
//    ml.load_train("saveResult20Epoc.csv");


    return 0;
}

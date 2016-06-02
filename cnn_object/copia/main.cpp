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
    ml.set_config(1, 0.1, 0.1); // (Epocas, Lrate CNN, Lrate MLP, Momentum)
    ml.set_image(28,1,0); // (Dimension, Channel, padding)
    ml.insert_layer(tLayer::convol,5,20,1);  // (convol, dim_filter, num_filter, jump)
    ml.insert_layer(tLayer::pool,2,2);      // (pool, dim_pool, jump)
    ml.insert_layer(tLayer::fullyC,100);
    ml.insert_layer(tLayer::fullyC,10);
    ml.struct_generate();
    ml.struct_initialize();
//    ml.load_train("PruebaMnist28X28-2.csv");
    ml.load_dataset(tData::train,"mnist_train_28x28.csv", 50000, 1);
    ml.train();
//    ml.save_train("PruebaMnist28X28-3.csv");
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
    
//    ml.set_config(1, 0.001, 0.001 ); // (Epocas, Lrate CNN, Lrate MLP)
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
////    ml.load_train("saveCifarNewB1.csv");
//    ml.load_dataset(tData::train,"cifarTrain.csv", 50000, 255);
//    ml.train();
//    ml.save_train("saveCifarNewB1.csv");
//    ml.load_dataset(tData::test, "cifarTest.csv", 10000, 255);
//    ml.test();



//  CIFAR-10 Opcion-2
    
//    ml.set_config(1, 0.7, 0.7 ); // (Epocas, Lrate CNN, Lrate MLP)
//    ml.set_image(32,3,0); // (Dimension, Channel, padding)
//    ml.insert_layer(tLayer::convol,5,16,1);  // (convol, dim_filter, num_filter, jump)
//    ml.insert_layer(tLayer::pool,2,2);      // (pool, dim_pool, jump)
//    ml.insert_layer(tLayer::convol,3,20,1);
//    ml.insert_layer(tLayer::pool,2,2);
//    ml.insert_layer(tLayer::fullyC,10);
//    ml.insert_layer(tLayer::fullyC,10);
//    ml.struct_generate();
//    ml.struct_initialize();
//    ml.load_dataset(tData::train,"cifarTrain.csv", 50000 , 255);
//    ml.train();
//    ml.load_dataset(tData::test, "cifarTest.csv", 10000, 255);
//    ml.test();

    return 0;
}

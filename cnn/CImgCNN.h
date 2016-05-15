//
//  CImgCNN.h
//  Mostrar Imagenes
//
//  Created by Andre Valdivia on 12/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//



#ifndef CImgCNN_h
#define CImgCNN_h

#include "CImg.h"

using namespace cimg_library;
using namespace std;

CImg<float> dibujar(double* data,int numMax,int x,int y,bool guardar,string nombre){
    static int a = 0;
    CImg<double> image(x,y);
    int j = 0;
    cimg_forXY(image, x, y){

        double dato = data[j]*255*255/numMax;
//        printf("(%d,%d) -> %f\n",x,y,dato);        
        image.draw_point(x, y, &dato);
        j++;
    }
    image.display();
    if (guardar == true) {
        string fileName = "Guardadas/" + nombre + ".bmp";
        image.save_bmp(fileName.c_str());
        a++;
    }
    return image;
}

#endif /* CImgCNN_h */


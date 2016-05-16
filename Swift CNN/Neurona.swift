//
//  Neurona.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 13/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class Neurona {
    var net:Double
    var out:Double
    var delta:Double
    var pesos = Array<Double>()
    init(numPesos:Int){
        net = 0
        out = 0
        delta = 0
        for _ in 0..<numPesos{
            pesos.append(Double.random0_1())
        }
    }
    
}
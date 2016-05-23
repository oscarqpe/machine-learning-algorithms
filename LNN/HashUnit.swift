//
//  HashVal.swift
//  LSH Test
//
//  Created by Andre Valdivia on 18/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation
import Darwin

class HashUnit{
    private var a = Array<Double>()
    private var b:Double
    private var key:Double = 0
    var delta:Double = 0
    init(){
        for _ in 0..<Param.lenInput{
            a.append(randGaussian())
        }
//        self.b = randUniform(0,rangeEnd: Double(Param.W))
        self.b = randGaussian()
    }
    
    //Retorna el key de la multiplicacion entre a*x+b
    func getKey(x:Array<Double>) -> Double{
        assert(x.count == a.count)
        let dP = Double(dotProduct(x, a))
        key = dP + b
        return key
    }
    
    func setDelta(delta:Double){
        self.delta = delta
    }
    
    func actualizarPesos(x:Array<Double>){
        for i in 0..<a.count{
            let val1 = Param.lrateLNN
            let val2 = self.delta
            let val3 = x[i]
            let valResult = val1 * val2 * val3
            a[i] = a[i] - (valResult)
        }
    }
}
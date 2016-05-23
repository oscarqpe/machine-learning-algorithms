//
//  Param.swift
//  LSH Test
//
//  Created by Andre Valdivia on 21/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

struct Param {
    static var W:Int = 0     //W =???? Supuestamente un parametro para normalizar el key del HashUnit
    static var K:Int = 0        //K = Numero de HashUnit en un Hash Table
    static var L:Int = 0       //L = Numero de HastTables en el Hash Layer
    static var lenInput:Int = 64
    static var hpos = Array<Int>()
    
    static var numNeuronasPorCapa = [3,8,11]
    static var lrate:Double = 0.1
    static var lrateLNN:Double = 10
    
    //    static var P1 = [1, 2, 5, 11, 17, 23, 31, 41, 47, 59 ]
    //    static var P2:Array<Int> = Param.primos(Param.K)
    static var P2 = Array<Int>()
    static func primos(n:Int) -> Array<Int>{
        var ret = Array<Int>()
        ret.append(1)
        var c = 1
        var p = 2
        var d = 2
        while c <= n{
            if(p % d == 0){
                if( p == d){
                    if (c % 2 == 1){
                        ret.append(p)

                    }
                    c++
                }
                d = 2
                p++
            }else{
                d++
            }
        }
        return ret
    }
    init(W:Int, K:Int, L:Int, lenInput:Int){
        Param.W = W
        Param.K = K
        Param.L = L
        Param.lenInput = lenInput
        Param.P2 = Param.primos(Param.K * 2 - 2)
        Param.hpos = Array<Int>(count: Param.K, repeatedValue: 0)
        for i in 0..<Param.K{
            if( i % 2 == 0){
                Param.hpos[i] = (i + 2) / 2 - 1
            }else{
                Param.hpos[i] = Param.K - ((i + 1) / 2)
            }
        }
        Param.numNeuronasPorCapa[0] = Param.L + 1
        //        Param.numNeuronasPorCapa.last! = Param
    }
}
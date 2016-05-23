//
//  HastTable.swift
//  LSH Test
//
//  Created by Andre Valdivia on 18/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class HashTable {
    private var hashUnits = Array<HashUnit>()
    var keys = Array<Double>()
    var key:Double = 0
    init(){
        for _ in 0..<Param.K{
            hashUnits.append(HashUnit())
            keys.append(0)
        }
    }
    
    //Retorna el key de multiplicar
    func train(x:Array<Double>) -> Double{
        
        //Get key de todos los hashUnits
        for i in 0..<hashUnits.count{
            keys[i] = hashUnits[i].getKey(x)
        }
        
        //Setear el key = keys * Primos
        self.key = 1
        for i in 0..<Param.P2.count{
            let pos = Param.hpos[i]
            self.key += keys[pos] * Double(Param.P2[i])
        }
        return key
    }
    
    //Setear el delta de todos los HashUnits
    func setDeltas(delta:Double){
        for i in 0..<Param.P2.count{
            let pos = Param.hpos[i]
            hashUnits[pos].setDelta(delta * Double(Param.P2[i])) // El delta del HT * Primo usado para hallar key
        }
    }
    
    func actualizarPesos(x:Array<Double>){
        for HU in hashUnits{
            HU.actualizarPesos(x)
        }
    }
}
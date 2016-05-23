//
//  LNN.swift
//  LSH Test
//
//  Created by Andre Valdivia on 18/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class LNN {
    var hashLayer = Array<HashTable>()
        var keys = Array<Double>()
    var mlp = MLP()
    var salida:String = ""
    init(){
        for _ in 0..<Param.L{
        hashLayer.append(HashTable())
        keys.append(0.0)
        }
    }
    
    private func ObtenerKeys(x:Array<Double>){
        //      Obtener todos el array de keys

        for i in 0..<hashLayer.count{
            keys[i] = hashLayer[i].train(x)
        }
    }
    
    func train(x: Array<Double>,salida:Int){
        
        //Forward
        //      Obtener todos el array de key
        ObtenerKeys(x)
        let max = normalizar(&keys)
        
//        self.salida = self.salida + String(keys)
//        self.salida += "\n\n"
        
        mlp.train(keys, salida: salida+1)
        
        //Hallar delta primera capa mlp
        for (index,HT) in hashLayer.enumerate(){
            HT.setDeltas(mlp.capas[0].neuronas[index + 1].delta / max)
        }
        
        //Actualizamos pesos
        for HT in hashLayer{
            HT.actualizarPesos(x)
        }
//        imprimirSalida(salida)

    }
    
    func test(x: Array<Double>,salida:Int) -> Bool {
        
        ObtenerKeys(x)
        let max = normalizar(&keys)
        var ret  = mlp.test(keys, salida: salida+1)
        return ret
    }
    
    private func imprimirSalida(salida:Int){
                var sal = Array<Double>(count: 11, repeatedValue: 0)
                for i in 0..<sal.count{
                    sal[i] = mlp.capas[2].neuronas[i].out
                }
        
                print(sal)
                print(salida)
                print(" ")
    }
    
    private func normalizar(inout array:Array<Double>) -> Double{
//        var max = 0.0
//        var min = 0.0
//        //Hallar el minimo
//        for val in array{
//            if min > val{
//                min = val
//            }
//        }
//        
//        //Sumarle a todos el mas negativo y hallar el maximo
//        for i in 0..<array.count{
//            array[i] = array[i] + abs(min)
//            if max < array[i]{
//                max = array[i]
//            }
//        }
//        
//        
//        for i in 0..<array.count{
//            array[i] = array[i]/max
//        }
//        
//        return max
        
        var max = 0.0
        for val in array{
            if max < abs(val){
                max = abs(val)
            }
        }
        
        for i in 0..<array.count{
            array[i] = array[i]/max
        }
        
        return max
    }
}
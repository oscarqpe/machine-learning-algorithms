//
//  MLP.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 13/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation
import Darwin

enum TypeACtivation{
    case euler
}


func activation(n:Double,type:TypeACtivation)->Double{
    return 1 / (1 + pow(exp(1.0), -n))
}

class MLP {
    var capas = Array<Capa>()
    var errores:Array<Double>
    var reales:Array<Double>
    init(){
        for (index,i) in Param.numNeuronasPorCapa.enumerate(){
            if index == 0 {
                capas.append(Capa(numNeuronas: i,numNeuronasCapaAnterior: 0))
            }
            else{
                capas.append(Capa(numNeuronas: i,numNeuronasCapaAnterior: Param.numNeuronasPorCapa[index-1]))
            }
        }
        errores = Array<Double>(count: Param.numNeuronasPorCapa.last!, repeatedValue: 0.0)
        reales = Array<Double>(count: Param.numNeuronasPorCapa.last!, repeatedValue: 0.0)
        
        //        Pesos ejemplo
                ejemplo(self)
        
    }
    func train(capEntrada:Array<Double>, salida:Int){
        
        //Asigamos valores en capa de entrada
        for (index,val) in capEntrada.enumerate(){
            capas[0].neuronas[index+1].out = val
        }
        
        //Asignamos valor real en reales
        for i in 1..<reales.count{
            if(i == salida){
                reales[i] = 0.99
            }else{
                reales[i] = 0.01
            }
        }
        
        //Forward
        for i in 1..<capas.count{
            capas[i].forward(capas[i-1])
        }
        
        //Hallar el error
        errores[0] = 0
        for i in 1..<errores.count{
            errores[i] = pow( reales[i] - (capas.last?.neuronas[i].out)! ,2) / 2
            errores[0] += errores[i]
        }
        
        //Backward
        //Capa ultima
        capas.last?.back(reales)
        
        //Otras capas
        for (var i = capas.count-2; i >= 0 ; i--){
            capas[i].back(capas[i+1])
        }
        
        //Actualizar pesos
        for(var i = capas.count-1 ; i > 0  ; i--){
            capas[i].actualizarPesos(capas[i-1])
        }
        
        
        
    }
}
//
//  Capa.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 13/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class Capa {
    var neuronas = Array<Neurona>()
    init(numNeuronas:Int,numNeuronasCapaAnterior:Int){
        for _ in 0..<numNeuronas{
            neuronas.append(Neurona(numPesos: numNeuronasCapaAnterior))
        }
        neuronas[0].out = 1
    }
    
    func forward(capaAnterior:Capa){
        for i in 1..<neuronas.count{
            let neu = neuronas[i]
            neu.net = 0
            for (index,neuAnt) in capaAnterior.neuronas.enumerate(){
                neu.net += neu.pesos[index] * neuAnt.out
            }
            neu.out = activation(neu.net, type: TypeACtivation.euler)
        }
    }
    
    func back(capaSiguiente:Capa){
        for i in 1..<neuronas.count{
            neuronas[i].delta = 0
            for j in 1..<capaSiguiente.neuronas.count{
                neuronas[i].delta += capaSiguiente.neuronas[j].delta * capaSiguiente.neuronas[j].pesos[i]
            }
            neuronas[i].delta *= neuronas[i].out * (1 - neuronas[i].out)
        }
    }
    
    func back(real:Array<Double>){
        for i in 1..<real.count{
            neuronas[i].delta = -(real[i] - neuronas[i].out) *  neuronas[i].out * (1 - neuronas[i].out)
        }
    }
    
    func actualizarPesos(capaAnterior:Capa){
        for i in 1..<neuronas.count{
            for(var index = 0 ; index < neuronas[i].pesos.count ; index++){
                neuronas[i].pesos[index] = neuronas[i].pesos[index] - Param.lrate * neuronas[i].delta * capaAnterior.neuronas[index].out
            }
        }
    }
    
}

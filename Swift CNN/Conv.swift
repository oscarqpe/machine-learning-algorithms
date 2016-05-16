//
//  Conv.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class Conv:Matrix{
    var convAnterior:Conv?
    var poolAnterior:Pool?
    var filtro:Filtro?
    var pad:Int
    init(rows: Int, columns: Int, id: Int, pad:Int = 0) {
        self.pad = pad
        super.init(rows: rows + pad*2, columns: columns + pad*2, id: id)
    }
    
    func connect(filtro:Filtro, convAnterior:Conv){
        self.filtro = filtro
        self.convAnterior = convAnterior
    }
    
    func connect(poolAnterior:Pool){
        self.poolAnterior = poolAnterior
    }
    
    func convul(){
        assert((convAnterior!.rows - self.rows) % filtro!.step == 0, "Error en convolucion, mala dimension de filtro o steps")
        //Recorremos la conv
        assert(filtro != nil,"Filtro es null")
        for r in 0..<self.rows{
            for c in 0..<self.columns{
                //Recorremos el filtro
                for r2 in 0..<filtro!.rows{
                    for c2 in 0..<filtro!.columns{
                        self[r,c] += filtro![r2,c2] * filtro![r2+(r*filtro!.step),c2+(c*filtro!.step)]
                    }
                }
            }
        }
        
    }
}
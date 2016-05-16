//
//  CapaConv.swift
//  Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class CapaConv:CapaCNN{
    var convoluciones = Array<Conv>()
    init(rows:Int, cols:Int,numConv:Int = 0,pad:Int = 0){
        super.init(rows: rows, cols: cols)
        for _ in 0..<numConv{
            convoluciones.append(Conv(rows: rows, columns: cols, id: id,pad: pad))
            self.id++
        }
    }
    
    func addMatrix(){

        convoluciones.append(Conv(rows: rows, columns: cols, id: id))
        self.id++
    }
    
    func connect(matrix: Int, filtro:Filtro, convAnterior:Conv){
            convoluciones[matrix].connect(filtro, convAnterior: convAnterior)
        self.id++
    }
    
    func addMatrix(m:Matrix){
        if(m.rows == rows && m.columns == cols){
            convoluciones.append(m as! Conv)
            self.id++
        }else{
            print("Error en addMatrix no coincide Rows o Cols")
        }
    }
    subscript() ->Array<Conv>{
        get{
            return convoluciones
        }
    }
}
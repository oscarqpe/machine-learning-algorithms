//
//  CapaFiltros.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class CapaFiltros {
    var filtros = Array<Filtro>()
    private var id:Int = 0
    private let rows:Int
    private let cols:Int
    private let step:Int
    init(rows:Int, cols:Int, step:Int, numFiltros:Int = 0){
        self.rows = rows
        self.cols = cols
        self.step = step
        for _ in 0..<numFiltros{
            filtros.append(Filtro(rows: rows, columns: cols, step: step, id: id))
            self.id++
        }
    }
    
    func addFiltro(){
        filtros.append(Filtro(rows: rows, columns: cols, step: step, id: id))
        self.id++
    }
    
    func addFiltro(filtro:Filtro){
        if(filtro.rows == rows && filtro.columns == cols){
        filtros.append(filtro)
        self.id++
        }
        else{
            print("Error agregando filtro falla de dimensiones")
        }
    }
    
    subscript (n:Int) -> Filtro{
        get{
            return filtros[n]
        }
        set(val) {
            filtros[n] = val
        }
    }
}
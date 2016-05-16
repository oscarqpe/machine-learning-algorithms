//
//  Filtro.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class Filtro: Matrix {
    let step:Int
    init(rows: Int, columns: Int,step:Int,id:Int) {
        self.step = step
        super.init(rows: rows, columns: columns, id:id)
        assert(rows%2 != 0, "Error al crear filtro, filtro par")
        assert(columns%2 != 0, "Error al crear filtro, filtro par")
        
        for i in 0..<grid.count{
            grid[i] = Double.random0_1()
        }
    }
}
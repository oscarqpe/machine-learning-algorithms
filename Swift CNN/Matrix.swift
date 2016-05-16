//
//  Matriz.swift
//  CNN Swift
//
//  Created by Andre Valdivia on 12/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation


//func conv(ini:Matrix, filter:Matrix,step:Int = 1) ->Matrix{
//    assert((ini.rows - filter.rows) % step == 0, "Error en convolucion, mala dimension de filtro o steps")
//    let dim:Int = (ini.rows - filter.rows)/step + 1
//    let ret = Matrix(rows: dim, columns: dim)
//    //Recorremos la conv
//    for r in 0..<ret.rows{
//        for c in 0..<ret.columns{
//            //Recorremos el filtro
//            for r2 in 0..<filter.rows{
//                for c2 in 0..<filter.columns{
//                    ret[r,c] += filter[r2,c2] * filter[r2+(r*step),c2+(c*step)]
//                }
//            }
//        }
//    }
//    return ret
//}

class Matrix{
    var id:Int
    var rows: Int, columns: Int
    internal var grid: [Double]
    init(rows: Int, columns: Int, id:Int) {
        self.id = id
        self.rows = rows
        self.columns = columns
        grid = Array(count: rows * columns, repeatedValue: 0.0)
    }
    private func indexIsValidForRow(row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
    subscript(row: Int, column: Int) -> Double {
        get {
            assert(indexIsValidForRow(row, column: column), "Index out of range")
            return grid[(row * columns) + column]
        }
        set {
            assert(indexIsValidForRow(row, column: column), "Index out of range")
            grid[(row * columns) + column] = newValue
        }
    }
    
    func mult(let matrix:Matrix) -> Double{
        var ret:Double = 0
        for r in 0..<self.rows{
            for c in 0..<self.columns{
                ret += self[r,c] * matrix[r,c]
            }
        }
        return ret
    }
}


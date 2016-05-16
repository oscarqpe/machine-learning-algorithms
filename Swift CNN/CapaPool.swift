//
//  CapaPool.swift
//  Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class CapaPool: CapaCNN {
    var pools = Array<Pool>()
    private let step:Int
    private let type:typePool
    init (rows:Int, cols:Int,step:Int,type:typePool, numPool:Int = 0){
        self.type = type
        self.step = step
        super.init(rows: rows, cols: cols)
        for _ in 0..<numPool{
            pools.append(Pool(rows: rows, columns: cols, step: step, id: id,type: type))
            self.id++
        }
    }
    
    func addMatrix(){
        pools.append(Pool(rows: rows, columns: cols, step: step, id: id, type: type))
        self.id++
    }
    
    func addPool(m:Pool){
        if(m.rows == rows && m.columns == cols){
            pools.append(m)
            self.id++
        }else{
            print("Error en addMatrix no coincide Rows o Cols")
        }
    }
    
    func  connect(matrix: Int, poolAnterior:Pool){
            pools[matrix].connect(poolAnterior)
    }
}
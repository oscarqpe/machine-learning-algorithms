//
//  Pool.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

enum typePool{
    case max
    case prom
}

class Pool:Matrix{
    var convAnterior:Conv?
    var poolAnterior:Pool?
    var step:Int
    var type:typePool
    init(rows: Int, columns: Int, step:Int, id:Int,type:typePool) {
        self.step = step
        self.type = type
        super.init(rows: rows, columns: columns, id:id)
        
    }
    
    func connect(convAnterior:Conv){
        self.convAnterior = convAnterior
    }
    
    func connect(poolAnterior:Pool){
        self.poolAnterior = poolAnterior
    }
    
    func pool(){
        switch type{
        case .max:
            for r in 0..<rows{
                for c in 0..<columns{
                    var max:Double = 0
                    for (var row2 = 0; row2 < convAnterior?.rows; row2 += step){
                        for (var col2 = 0; col2 < convAnterior?.columns; col2 += step){
                            if(convAnterior![row2,col2] > max){
                                max = convAnterior![row2,col2]
                            }
                        }
                    }
                    self[r,c] = max
                }
            }
        
        case .prom:
            
            for r in 0..<rows{
                for c in 0..<columns{
                    var prom:Double = 0
                    var count:Int = 0
                    for (var row2 = 0; row2 < convAnterior?.rows; row2 += step){
                        for (var col2 = 0; col2 < convAnterior?.columns; col2 += step){
                            prom += convAnterior![row2,col2]
                            count++
                        }
                    }
                    self[r,c] = prom
                }
            }
        }
        
    }
}
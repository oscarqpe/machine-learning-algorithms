//
//  CNN.swift
//  CNN Swift
//
//  Created by Andre Valdivia on 12/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

enum typeMatrix{
    case Filter
    case Conv
    case Pool
}

class CNN {
    private var capas = Array<CapaCNN>()
    private var filtros = Array<CapaFiltros>()
    var clase:Int?
    init(param:Array<Array<AnyObject>>){
        
    }
    
    init(capasCNN:Array<CapaCNN>, capasFiltros:Array<CapaFiltros>){
        self.capas = capasCNN
        self.filtros = capasFiltros
    }
    
    init(rowsInput:Int, colsInput:Int,pad:Int = 0){
        capas.append(CapaConv(rows:rowsInput,cols: colsInput,pad: pad))
    }
    
    func addCapaConv(rows:Int, cols:Int, numConv:Int = 0){
        let c = CapaConv(rows: rows, cols: cols)
        capas.last!.CapaSiguiente = c
        c.CapaAnterior = capas.last!
        capas.append(c)
        
    }
    
    func addCapaPool(rows:Int, cols:Int, step:Int,type:typePool, numPool:Int = 0 ){
        let c = CapaPool(rows: rows, cols: cols, step: step, type: type, numPool: numPool)
        capas.last!.CapaSiguiente = c
        c.CapaAnterior = capas.last!
        capas.append(c)
    }
    
    func addCapaFiltro(rows:Int, cols:Int, step:Int, numFiltros:Int = 0){
        filtros.append(CapaFiltros(rows: rows, cols: cols, step: step, numFiltros: numFiltros))
    }
    
    func train(){
        
        for c in capas{
            if let capa = c as? CapaConv{
                for conv in capa.convoluciones{
                    conv.convul()
                }
            }else if let capa = c as? CapaPool{
                for pool in capa.pools{
                    pool.pool()
                }
            }
        }
    }
}






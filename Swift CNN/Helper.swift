//
//  Helper.swift
//  CNN Swift
//
//  Created by Andre Valdivia on 13/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

public extension Double{
    static func random0_1() ->Double{
//                return Double(arc4random_uniform(UInt32(upper - lower))) + lower
        return Double(arc4random()) / Double(UINT32_MAX)
    }
}
struct mmm {
    var i:Array<Double> = [ 0.05, 0.10 ]
}

func ejemplo(m:MLP){
    m.capas[1].neuronas[1].pesos[0] = 0.35
    m.capas[1].neuronas[1].pesos[1] = 0.15
    m.capas[1].neuronas[1].pesos[2] = 0.20
    m.capas[1].neuronas[2].pesos[0] = 0.35
    m.capas[1].neuronas[2].pesos[1] = 0.25
    m.capas[1].neuronas[2].pesos[2] = 0.30
    
    m.capas[2].neuronas[1].pesos[0] = 0.60
    m.capas[2].neuronas[1].pesos[1] = 0.40
    m.capas[2].neuronas[1].pesos[2] = 0.45
    m.capas[2].neuronas[2].pesos[0] = 0.60
    m.capas[2].neuronas[2].pesos[1] = 0.50
    m.capas[2].neuronas[2].pesos[2] = 0.55
}


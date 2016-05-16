//
//  CapaCNN.swift
//  Pruebas Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

class CapaCNN {
    var CapaAnterior:CapaCNN?
    var CapaSiguiente:CapaCNN?
    var id:Int = 0
    internal let rows:Int
    internal let cols:Int
    init(rows:Int, cols:Int){
        self.rows = rows
        self.cols = cols
    }
//    subscript(row: Int) -> Matrix {
//        get {
//            switch type{
//            case .Conv:
//                return convoluciones![row]
//            case .Pool:
//                return pools![row]
//            default:
//                print("Hubo un error en subscript CNN")
//                return convoluciones![row]                
//            }
//        }
//        set(va) {
//            switch type{
//            case .Conv:
//                convoluciones![row] = va as! Conv
//            case .Pool:
//                pools![row] = va as! Pool
//            default:
//                print("Hubo un error en subscript CNN")
//                convoluciones![row] = va as! Conv
//            }
//        }
//    }
//    func train(){
//        switch type{
//        case .Conv:
//            for m in convoluciones!{
//                m.convul()
//            }
//        case .Pool:
//            for m in pools!{
//                m.pool()
//            }
//        default:
//            print("Error en train")
//
//        }
//    }
}
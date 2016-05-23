//
//  Helper.swift
//  LSH Test
//
//  Created by Andre Valdivia on 18/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation
import Darwin
import GameplayKit

public extension Double{
    static func random0_1() ->Double{
        return Double(arc4random()) / Double(UINT32_MAX)
    }
}

//func randNormal() ->Double{
//    let random = GKRandomSource()
//    let sigma:Float = Float(2) / Float(Param.lenInput)
//    let ret = GKGaussianDistribution(randomSource: random, mean: 0, deviation: 0.1)
//    let r = ret.nextUniform()
//    return Double(ret.nextUniform())
//}

func randGaussian() -> Double{
    var x1:Double, x2:Double;
    repeat
    {
        x1 = randUniform(0.0, rangeEnd: 1.0)
    } while (x1 == 0); // cannot take log of 0.
    x2 = randUniform(0.0, rangeEnd: 1.0)
    var z:Double
    z = Double(sqrt(-2.0 * log(x1)) * cos(2.0 * M_PI * x2))
    return z;
}

func randUniform(rangeStart:UInt32,rangeEnd:UInt32) -> Int{
    var r:UInt32;
    r = rangeStart + arc4random_uniform((rangeEnd - rangeStart) + 1);
    return Int(r);
}

func randUniform(rangeStart:Double, rangeEnd:Double) ->Double{
    var r:Double
    r = rangeStart + ((rangeEnd - rangeStart) * Double.random0_1());
    return r
}

func dotProduct(x1:Array<Double>,_ x2:Array<Double>) -> Double{
    var ret:Double = 0
    for (var i = 0 ; i < x1.count ; i++){
        ret += x1[i] * x2[i]
    }
    return ret
}


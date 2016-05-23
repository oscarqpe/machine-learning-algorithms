//
//  main.swift
//  LSH Test
//
//  Created by Andre Valdivia on 18/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

var data = Data()
let pathTrain = NSBundle.mainBundle().pathForResource("mnist", ofType:"csv")
let fileContentTrain = try? NSString(contentsOfFile: pathTrain!, encoding: NSUTF8StringEncoding)
data.insertDataTrain(fileContentTrain! as String)
let pathTest = NSBundle.mainBundle().pathForResource("mnist_test", ofType:"csv")
let fileContentTest = try? NSString(contentsOfFile: pathTest!, encoding: NSUTF8StringEncoding)
data.insertDataTest(fileContentTest! as String)


var p = Param(W: 10, K: 12, L: 150, lenInput: 64)

var lnn = LNN()

//var dat2 = data.readTrain()

for _ in 0..<10000{
    var dat = data.readTrain()
//    if(dat.0 == 0){
        lnn.train(dat.1,salida: dat.0)
//    }
}

//writeData(lnn.salida,nameFile: "Keys LSH")







var aciertos:Double = 0
var desaciertos:Double = 0
for dat in data.test{
    var resp = lnn.test(dat.data, salida: dat.clase)
    if(resp == true){
        aciertos++
    }else{
        desaciertos++
    }
}
print(aciertos)
print(desaciertos)
print("Porcentaje: \(aciertos/(aciertos+desaciertos) * 100)%")

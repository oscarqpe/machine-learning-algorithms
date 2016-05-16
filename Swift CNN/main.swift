//
//  main.swift
//  Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

print("Hello, World!")

var data = Data()

//Read data
let pathTrain = NSBundle.mainBundle().pathForResource("mnist", ofType:"csv")
let fileContentTrain = try? NSString(contentsOfFile: pathTrain!, encoding: NSUTF8StringEncoding)
data.readDataTrain(fileContentTrain! as String)
let pathTest = NSBundle.mainBundle().pathForResource("mnist_test", ofType:"csv")
let fileContentTest = try? NSString(contentsOfFile: pathTrain!, encoding: NSUTF8StringEncoding)
data.readDataTrain(fileContentTest! as String)



var cnn = CNN(rowsInput: 6, colsInput: 6)
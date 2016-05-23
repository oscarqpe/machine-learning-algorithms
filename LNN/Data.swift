//
//  Data.swift
//  Swift CNN
//
//  Created by Andre Valdivia on 15/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation

struct oneData{
    var clase:Int
    var data = [Double]()
}
class Data {
    var test = Array<oneData>()
    var train = Array<oneData>()
    
    private var counterTrain = -1
    
    func insertTrain(clase:Int,data:Array<Double>){
        train.append(oneData(clase: clase, data: data))
    }
    
    func insertTest(clase:Int,data:Array<Double>){
        test.append(oneData(clase: clase, data: data))
    }
    
    func readTrain() -> (Int,[Double]){

        if(counterTrain >= train.count-1){
            counterTrain = 0
        }else{
            counterTrain++
        }
        return (train[counterTrain].clase,train[counterTrain].data)
    }
    
    func insertDataTrain(file:String,indexClass:Int = 0){
        let dataLine = file.componentsSeparatedByString("\n")
        for line in dataLine{
            if line != ""{
                let arrayLine = line.componentsSeparatedByString(";")
                var classMember:Int?
                var dataArray = [Double]()
                for (index,unitData) in arrayLine.enumerate(){
                    if(index == indexClass){
                        
                        classMember = Int(unitData)!
                    }else{
                        dataArray.append(Double(unitData)!/17)
                    }
                }
                insertTrain(classMember!, data: dataArray)
            }
        }
    }
    
    func insertDataTest(file:String,indexClass:Int = 0){
        let dataLine = file.componentsSeparatedByString("\n")
        for line in dataLine{
            if line != ""{
                let arrayLine = line.componentsSeparatedByString(";")
                var classMember:Int?
                var dataArray = [Double]()
                for (index,unitData) in arrayLine.enumerate(){
                    if(index == indexClass){
                        
                        classMember = Int(unitData)!
                    }else{
                        dataArray.append(Double(unitData)! / 17)
                    }
                }
                insertTest(classMember!, data: dataArray)
            }
        }
    }

}
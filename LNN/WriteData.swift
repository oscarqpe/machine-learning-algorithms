//
//  WriteData.swift
//  LSH Test
//
//  Created by Andre Valdivia on 22/05/16.
//  Copyright © 2016 Andre Valdivia. All rights reserved.
//

import Foundation
func writeData(text:String, nameFile:String){
    let file = nameFile //this is the file. we will write to and read from it

//    let text = text //just a text

    if let dir = NSSearchPathForDirectoriesInDomains(NSSearchPathDirectory.DocumentDirectory, NSSearchPathDomainMask.AllDomainsMask, true).first {
        let path = NSURL(fileURLWithPath: dir).URLByAppendingPathComponent(file)
        
        //writing
        do {
            try text.writeToURL(path, atomically: false, encoding: NSUTF8StringEncoding)
        }
        catch {/* error handling here */}
        
//        //reading
//        do {
//            let text2 = try NSString(contentsOfURL: path, encoding: NSUTF8StringEncoding)
//        }
//        catch {/* error handling here */}
    }
}
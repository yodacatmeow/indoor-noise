//
//  fft.swift
//  hello-world
//
//  Created by yodacat on 01/10/2019.
//  Copyright © 2019 yodacat. All rights reserved.
//

import Foundation
import AVFoundation
import Accelerate

func audioPathToFloatArray(filepath:URL, fs:Double, channels:UInt32)->Array<Float>?{
    do{
        let file = try AVAudioFile(forReading:filepath)
        let sampleLength = file.length
        let format = AVAudioFormat(commonFormat: AVAudioCommonFormat.pcmFormatFloat32,
                                   sampleRate: fs,
                                   channels: channels,
                                   interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleLength))!
        try! file.read(into:buffer)
        var floatArray = Array(UnsafeBufferPointer(start: buffer.floatChannelData?[0], count:Int(buffer.frameLength)))
        return floatArray
    }
    catch{
        print("[audioPathToFloatArrau] failed..")
        return nil
    }
}

func fftForward(inputSignal:Array<Float>, nFFT:Int, hanningWindow:Bool, destroyFFT:Bool) -> Array<Float>?{
    let n = nFFT
    let halfN = Int(n/2)
    let signalCount = inputSignal.count
    
    var transferBuffer = [Float](repeating: 0, count: signalCount)
    
    // Windowing
    if(hanningWindow == true){
        var window = [Float](repeating: 0, count: signalCount)
        vDSP_hann_window(&window, vDSP_Length(signalCount), Int32(vDSP_HANN_NORM))
        vDSP_vmul(inputSignal, 1, window,
                  1, &transferBuffer, 1, vDSP_Length(signalCount))
    }
    else{
        transferBuffer = inputSignal
    }
    
    let observed: [DSPComplex] = stride(from: 0, to: Int(n), by: 2).map {
        return DSPComplex(real: transferBuffer[$0],
                          imag: transferBuffer[$0.advanced(by: 1)])
    }
        
    var forwardInputReal = [Float](repeating: 0, count: halfN)
    var forwardInputImag = [Float](repeating: 0, count: halfN)
    var forwardInput = DSPSplitComplex(realp: &forwardInputReal,
                                       imagp: &forwardInputImag)
    
    vDSP_ctoz(observed, 2,
              &forwardInput, 1,
              vDSP_Length(halfN))
    
    
    let log2n = vDSP_Length(log2(Float(n)))
    
    guard let fftSetUp = vDSP_create_fftsetup(
        log2n,
        FFTRadix(kFFTRadix2)) else {
            fatalError("Can't create FFT setup.")
    }
    
    var forwardOutputReal = [Float](repeating: 0, count: halfN)
    var forwardOutputImag = [Float](repeating: 0, count: halfN)
    var forwardOutput = DSPSplitComplex(realp: &forwardOutputReal,
                                        imagp: &forwardOutputImag)
    
    vDSP_fft_zrop(fftSetUp,
                  &forwardInput, 1,
                  &forwardOutput, 1,
                  log2n,
                  FFTDirection(kFFTDirection_Forward))
    
    
    // Magnitude
    var magnitudes = [Float](repeating: 0.0, count: halfN)
    vDSP_zvmags(&forwardOutput, 1, &magnitudes, 1, vDSP_Length(halfN))
    
    // Normalization???????????????????
    
    // Destroy "fftsetup"
    if (destroyFFT == true) {
        
        defer {
            vDSP_destroy_fftsetup(fftSetUp)
        }
    }

    //return forwardOutputReal
    return magnitudes
}

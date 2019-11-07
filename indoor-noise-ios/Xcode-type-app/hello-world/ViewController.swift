//
//  ViewController.swift
//  hello-world
//
//  Created by yodacat on 19/08/2019.
//  Copyright © 2019 yodacat. All rights reserved.
//

import UIKit
import AVFoundation
import Accelerate       // vDSP
import CoreML

class ViewController: UIViewController, AVAudioRecorderDelegate, UITableViewDelegate, UITableViewDataSource {
    
    // Two new variables
    var recordingSession:AVAudioSession!
    var audioRecorder:AVAudioRecorder!
    var audioPlayer:AVAudioPlayer!
    
    
    // Number of Records
    var numberOfRecords:Int = 0

    @IBOutlet weak var recordLabel: UIButton!
    @IBOutlet weak var myTableView: UITableView!
    
    @IBOutlet weak var classificationResult: UILabel!
    
    
    // "Record button" (action)
    @IBAction func record(_ sender: Any) {
        // Check if we have an active recorder
        if audioRecorder == nil{
            // if we record multiple records
            numberOfRecords += 1
            // Give name using the function that we already defined; via appending (name of file name will be ~.m4a)
            let filename = getDirectory().appendingPathComponent("\(numberOfRecords).m4a")
            
            // Array containing settings
            let settings = [AVFormatIDKey: Int(kAudioFormatMPEG4AAC),AVSampleRateKey:16000, AVNumberOfChannelsKey:1, AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue]
            
            // Start audio recording
            do{
                audioRecorder = try AVAudioRecorder(url:filename,settings: settings)
                audioRecorder.delegate = self
                audioRecorder.record()
                
                // Change the button label
                recordLabel.setTitle("Stop Recording", for:.normal)
                
            
            }
            // If that doesn't work
            catch{
                displayAlert(title: "Ups", message: "recording failed")
            }
        }
        else{
            // Stopping recording
            audioRecorder.stop()
            audioRecorder=nil
            
            // User default
            UserDefaults.standard.set(numberOfRecords, forKey: "myNumber")
            // Refresh our TABLE VIEW
            myTableView.reloadData()
            
            recordLabel.setTitle("Start Recording", for:.normal)
        

            
        }
    }
    

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        // Setting up a Session
        recordingSession = AVAudioSession.sharedInstance()  // feed something
        
        // Cast as integer
        if let number:Int = UserDefaults.standard.object(forKey: "myNumber") as? Int{
            numberOfRecords = number
        }
        
        // Ask the permission
        AVAudioSession.sharedInstance().requestRecordPermission{ (haspermission) in
            if haspermission{
                print("Microphone access accepted!")
            }
            
        }
    }
    
    // Function that gets path to directory
    func getDirectory() -> URL{
        // search for all the directory in our "documentDirectory"
        let paths = FileManager.default.urls(for:.documentDirectory, in:.userDomainMask)
        // Take the first one as our path
        let documentDirectory = paths[0]
        // return URL to the documentDirectory
        return documentDirectory
    }

    // Function that displays an alert with two arguments
    func displayAlert(title:String, message: String){
        // take title and messages
        let alert = UIAlertController(title:title, message: message, preferredStyle: .alert)
        // Action
        alert.addAction(UIAlertAction(title: "dismiss", style:.default, handler:nil))
        // Present
        present(alert, animated:true, completion: nil)
    }
    
    // Setting up TABLE VIEW
    func tableView(_ tableView: UITableView, numberOfRowsInSection section:Int) -> Int {
        return numberOfRecords
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        cell.textLabel?.text = String(indexPath.row + 1)
        return cell
    }
    
    //
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath){
        let path = getDirectory().appendingPathComponent("\(indexPath.row + 1).m4a")
        
        do{
            // Signal to a float array using "signal.swift"
            var signal = audioPathToFloatArray(filepath:path, fs:16000.0, channels:1)!
            
            // Test of a signal to freqeuncy magnitude using "signal.swift"
            //let fftOut = fftForward(inputSignal:signal, nFFT:2048, hanningWindow: true, destroyFFT: true)
            //print(fftOut)
            
           
            let model = my_sound_classifier()
            
            let file = try AVAudioFile(forReading:path)
            let sampleLength = file.length
            let format = AVAudioFormat(commonFormat: AVAudioCommonFormat.pcmFormatFloat32,
                                       sampleRate: 16000,
                                       channels: 1,
                                       interleaved: false)!
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(sampleLength))!
            try! file.read(into:buffer)
            var bufferData = Array(UnsafeBufferPointer(start: buffer.floatChannelData?[0], count:Int(buffer.frameLength)))
            
            // Chunk data and set to CoreML model
            let windowSize = 15600
            guard let audioData = try? MLMultiArray(shape:[windowSize as NSNumber],
                                                    dataType:MLMultiArrayDataType.float32)
                else {
                    fatalError("Can not create MLMultiArray")
            }
            
            var results = [Dictionary<String, Double>]()
            let frameLength = Int(buffer.frameLength)
            var audioDataIndex = 0
            
            // Iterate over all the samples, chunking calls to analyze every 15600
            for i in 0..<frameLength {
                audioData[audioDataIndex] = NSNumber.init(value: bufferData[i])
                if audioDataIndex >= windowSize {
                    let modelInput = my_sound_classifierInput(audio: audioData)
                    
                    guard let modelOutput = try? model.prediction(input: modelInput) else {
                        fatalError("Error calling predict")
                    }
                    results.append(modelOutput.typeProbability)
                    audioDataIndex = 0
                } else {
                    audioDataIndex += 1
                }
            }
            
            // Handle remainder by passing with zero
            if audioDataIndex > 0 {
                for audioDataIndex in audioDataIndex...windowSize {
                    audioData[audioDataIndex] = 0
                }
                let modelInput = my_sound_classifierInput(audio: audioData)
                
                guard let modelOutput = try? model.prediction(input: modelInput) else {
                    fatalError("Error calling predict")
                }
                results.append(modelOutput.typeProbability)
            }
            let highestCategory = results[0].max{a,b in a.value < b.value}
            print(highestCategory?.key)
            print(results)
            
            classificationResult.text = highestCategory?.key
            
            //print(type(of: highestCategory))
            
            audioPlayer = try AVAudioPlayer(contentsOf: path)
            audioPlayer.play()
            
            
        }
            
        catch{
            
        }
        
    }
    
    
}

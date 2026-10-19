import AVFoundation
import CoreML
import Foundation
import ImageIO
import Vision

struct PoseDetectionError: Error {
    let message: String
    init(_ message: String) { self.message = message }
}

/// Native side of `pyxpo_pose_detection`: Vision `VNDetectHumanBodyPoseRequest`. Work runs on one serial queue.
final class PoseDetectionModule: PxModule {
    static let moduleName = "PoseDetection"
    static let queue = DispatchQueue(label: "pyxpo.posedetection", qos: .userInitiated)

    init() {}

    func call(_ ctx: PxModuleContext, _ method: String, _ args: [Any], _ reply: @escaping PxReply) {
        switch method {
        case "available":
            run(reply) { PoseDetect.works }
        case "detectImage":
            run(reply) { try PoseDetect.image(try Self.str(args, 0)) }
        case "detectVideo":
            let o = (args.count > 1 ? args[1] as? [String: Any] : nil) ?? [:]
            let id = (o["id"] as? NSNumber)?.intValue ?? 0
            run(reply) { [weak ctx] in
                var last = -1
                return try PoseDetect.video(try Self.str(args, 0), fps: (o["fps"] as? NSNumber)?.doubleValue ?? 15,
                                            maxDuration: (o["maxDuration"] as? NSNumber)?.doubleValue ?? 30,
                                            maxSize: (o["maxSize"] as? NSNumber)?.intValue ?? 640) { p in
                    let pct = Int(p * 100)
                    if pct != last { last = pct; ctx?.emit("PoseDetection.progress", ["id": id, "progress": p]) }
                }
            }
        default:
            reply(nil, "unknown native call PoseDetection.\(method)")
        }
    }

    static func str(_ args: [Any], _ i: Int) throws -> String {
        guard i < args.count, let s = args[i] as? String else { throw PoseDetectionError("missing string argument \(i)") }
        return s
    }

    private func run(_ reply: @escaping PxReply, _ work: @escaping () throws -> Any?) {
        Self.queue.async {
            var value: Any?
            var message: String?
            do { value = try work() }
            catch let e as PoseDetectionError { message = e.message }
            catch { message = "\(error)" }
            reply(value, message)
        }
    }
}

enum PoseDetect {
    // Vision joint -> COCO name (the Python side's LANDMARKS).
    static let names: [(VNHumanBodyPoseObservation.JointName, String)] = [
        (.nose, "nose"), (.leftEye, "left_eye"), (.rightEye, "right_eye"), (.leftEar, "left_ear"), (.rightEar, "right_ear"),
        (.leftShoulder, "left_shoulder"), (.rightShoulder, "right_shoulder"), (.leftElbow, "left_elbow"),
        (.rightElbow, "right_elbow"), (.leftWrist, "left_wrist"), (.rightWrist, "right_wrist"), (.leftHip, "left_hip"),
        (.rightHip, "right_hip"), (.leftKnee, "left_knee"), (.rightKnee, "right_knee"), (.leftAnkle, "left_ankle"),
        (.rightAnkle, "right_ankle"),
    ]

    static func url(_ uri: String) -> URL {
        uri.hasPrefix("file://") ? (URL(string: uri) ?? URL(fileURLWithPath: String(uri.dropFirst(7)))) : URL(fileURLWithPath: uri)
    }

    /// False when Vision cannot run the request here. Recent iOS simulators fail every body pose request with
    /// code 9 ("Unable to setup request"), so the simulator tries one blank picture first; a phone always can.
    static let works: Bool = {
        #if targetEnvironment(simulator)
        guard let ctx = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let blank = ctx.makeImage() else { return false }
        return (try? detect(blank)) != nil
        #else
        return true
        #endif
    }()

    /// Runs every stage of `request` on the CPU.
    static func onCPU(_ request: VNRequest) {
        if #available(iOS 17.0, *) {
            guard let stages = try? request.supportedComputeStageDevices else { return }
            for (stage, devices) in stages {
                for device in devices {
                    if case .cpu = device {
                        request.setComputeDevice(device, for: stage)
                        break
                    }
                }
            }
        } else {
            request.usesCPUOnly = true
        }
    }

    /// Landmarks of the most confident person in an upright image, in its pixels (y down).
    static func detect(_ image: CGImage) throws -> [String: [Double]] {
        let request = VNDetectHumanBodyPoseRequest()
        #if targetEnvironment(simulator)
        onCPU(request)  // the Simulator has no Neural Engine
        #endif
        let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
        try handler.perform([request])
        guard let people = request.results, !people.isEmpty else { return [:] }
        let best = people.max { $0.confidence < $1.confidence } ?? people[0]
        let points = (try? best.recognizedPoints(.all)) ?? [:]
        let w = Double(image.width), h = Double(image.height)
        var out: [String: [Double]] = [:]
        for (joint, label) in names {
            guard let p = points[joint], p.confidence > 0 else { continue }
            out[label] = [Double(p.location.x) * w, (1 - Double(p.location.y)) * h, Double(p.confidence)]
        }
        return out
    }

    /// A photo with its EXIF orientation applied, longest side at most 2048.
    static func upright(_ uri: String) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url(uri) as CFURL, nil) else {
            throw PoseDetectionError("cannot read image: \(url(uri).lastPathComponent)")
        }
        let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                     kCGImageSourceCreateThumbnailWithTransform: true,
                                     kCGImageSourceThumbnailMaxPixelSize: 2048]
        guard let img = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else {
            throw PoseDetectionError("cannot read image: \(url(uri).lastPathComponent)")
        }
        return img
    }

    static func image(_ uri: String) throws -> [String: Any] {
        let img = try upright(uri)
        return ["width": img.width, "height": img.height, "landmarks": try detect(img)]
    }

    static func video(_ uri: String, fps: Double, maxDuration: Double, maxSize: Int,
                      progress: (Double) -> Void) throws -> [String: Any] {
        let asset = AVURLAsset(url: url(uri))
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw PoseDetectionError("the file has no video track: \(url(uri).lastPathComponent)")
        }
        let length = min(CMTimeGetSeconds(asset.duration), maxDuration)
        guard length.isFinite, length > 0 else { throw PoseDetectionError("cannot read video: \(url(uri).lastPathComponent)") }
        let size = track.naturalSize.applying(track.preferredTransform)
        let upW = abs(Double(size.width)), upH = abs(Double(size.height))
        let k = max(upW, upH) > Double(maxSize) ? Double(maxSize) / max(upW, upH) : 1
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: (upW * k).rounded(), height: (upH * k).rounded())
        let tolerance = CMTime(seconds: 0.5 / fps, preferredTimescale: 600)
        gen.requestedTimeToleranceBefore = tolerance
        gen.requestedTimeToleranceAfter = tolerance
        let count = max(1, Int(length * fps))
        var frames: [[String: Any]] = []
        var outW = 0, outH = 0
        for i in 0..<count {
            let t = Double(i) / fps
            try autoreleasepool {
                if let img = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) {
                    outW = img.width; outH = img.height
                    frames.append(["t": t, "landmarks": try detect(img)])
                }
            }
            progress(Double(i + 1) / Double(count))
        }
        if frames.isEmpty { throw PoseDetectionError("could not read any frame of the video") }
        return ["width": outW, "height": outH, "duration": length, "frames": frames]
    }
}

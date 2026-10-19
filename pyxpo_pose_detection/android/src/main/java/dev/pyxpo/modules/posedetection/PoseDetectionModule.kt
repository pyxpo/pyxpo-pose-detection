package dev.pyxpo.modules.posedetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import dev.pyxpo.moduleapi.PxModule
import dev.pyxpo.moduleapi.PxModuleContext
import dev.pyxpo.moduleapi.PxReply
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.roundToLong

class PoseDetectionError(message: String) : Exception(message)

/** Native side of `pyxpo_pose_detection`: ML Kit pose detection (bundled model). Work runs on one background thread. */
class PoseDetectionModule : PxModule() {
    override val name = "PoseDetection"

    override fun call(ctx: PxModuleContext, method: String, args: JSONArray, reply: PxReply) {
        val app = ctx.activity.applicationContext
        when (method) {
            "available" -> background(reply) { true }
            "detectImage" -> background(reply) { PoseDetect.image(app, str(args, 0)) }
            "detectVideo" -> background(reply) {
                val o = args.optJSONObject(1) ?: JSONObject()
                val id = o.optInt("id")
                var last = -1
                PoseDetect.video(app, str(args, 0), o.optDouble("fps", 15.0), o.optDouble("maxDuration", 30.0),
                    o.optInt("maxSize", 640)) { p ->
                    val pct = (p * 100).toInt()
                    if (pct != last) { last = pct; ctx.emit("PoseDetection.progress", mapOf("id" to id, "progress" to p)) }
                }
            }
            else -> reply(null, "unknown native call PoseDetection.$method")
        }
    }

    companion object {
        private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "pyxpo-posedetection") }

        fun str(args: JSONArray, i: Int): String {
            if (i >= args.length() || args.isNull(i)) throw PoseDetectionError("missing string argument $i")
            return args.getString(i)
        }

        private fun background(reply: PxReply, work: () -> Any?) {
            executor.execute {
                var value: Any? = null
                var message: String? = null
                try { value = work() }
                catch (e: PoseDetectionError) { message = e.message }
                catch (e: Throwable) { message = e.toString() }
                reply(value, message)
            }
        }
    }
}

/** Decoding, frame sampling and the ML Kit calls. Only touched on the module's executor. */
object PoseDetect {
    // ML Kit landmark -> COCO name (the Python side's LANDMARKS).
    private val NAMES = linkedMapOf(
        PoseLandmark.NOSE to "nose", PoseLandmark.LEFT_EYE to "left_eye", PoseLandmark.RIGHT_EYE to "right_eye",
        PoseLandmark.LEFT_EAR to "left_ear", PoseLandmark.RIGHT_EAR to "right_ear",
        PoseLandmark.LEFT_SHOULDER to "left_shoulder", PoseLandmark.RIGHT_SHOULDER to "right_shoulder",
        PoseLandmark.LEFT_ELBOW to "left_elbow", PoseLandmark.RIGHT_ELBOW to "right_elbow",
        PoseLandmark.LEFT_WRIST to "left_wrist", PoseLandmark.RIGHT_WRIST to "right_wrist",
        PoseLandmark.LEFT_HIP to "left_hip", PoseLandmark.RIGHT_HIP to "right_hip",
        PoseLandmark.LEFT_KNEE to "left_knee", PoseLandmark.RIGHT_KNEE to "right_knee",
        PoseLandmark.LEFT_ANKLE to "left_ankle", PoseLandmark.RIGHT_ANKLE to "right_ankle",
    )

    private var single: PoseDetector? = null

    private fun singleDetector(): PoseDetector = single ?: PoseDetection.getClient(
        PoseDetectorOptions.Builder().setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE).build()
    ).also { single = it }

    private fun landmarks(pose: Pose): Map<String, List<Double>> {
        val out = LinkedHashMap<String, List<Double>>()
        for ((type, label) in NAMES) {
            val lm = pose.getPoseLandmark(type) ?: continue
            out[label] = listOf(lm.position.x.toDouble(), lm.position.y.toDouble(), lm.inFrameLikelihood.toDouble())
        }
        return out
    }

    private fun detect(detector: PoseDetector, bmp: Bitmap): Map<String, List<Double>> =
        landmarks(Tasks.await(detector.process(InputImage.fromBitmap(bmp, 0))))

    private fun isContent(uri: String) = uri.startsWith("content://")

    private fun path(uri: String): String =
        if (uri.startsWith("file://")) Uri.parse(uri).path ?: uri.substring(7) else uri

    private fun rotate(bmp: Bitmap, degrees: Int, flip: Boolean = false): Bitmap {
        if (degrees == 0 && !flip) return bmp
        val m = Matrix()
        if (flip) m.postScale(-1f, 1f)
        m.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun scaled(bmp: Bitmap, maxSize: Int): Bitmap {
        val edge = max(bmp.width, bmp.height)
        if (edge <= maxSize) return bmp
        val k = maxSize.toFloat() / edge
        return Bitmap.createScaledBitmap(bmp, max(1, (bmp.width * k).roundToInt()), max(1, (bmp.height * k).roundToInt()), true)
    }

    /** A photo, EXIF orientation applied, longest side at most 2048. */
    private fun decodeImage(ctx: Context, uri: String): Bitmap {
        val bmp: Bitmap?
        var exif: ExifInterface? = null
        if (isContent(uri)) {
            val u = Uri.parse(uri)
            bmp = ctx.contentResolver.openInputStream(u)?.use { BitmapFactory.decodeStream(it) }
            exif = try { ctx.contentResolver.openInputStream(u)?.use { ExifInterface(it) } } catch (e: Exception) { null }
        } else {
            val p = path(uri)
            bmp = BitmapFactory.decodeFile(p)
            exif = try { ExifInterface(p) } catch (e: Exception) { null }
        }
        if (bmp == null) throw PoseDetectionError("cannot read image: ${uri.substringAfterLast('/')}")
        val up = when (exif?.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> rotate(bmp, 90)
            ExifInterface.ORIENTATION_ROTATE_180 -> rotate(bmp, 180)
            ExifInterface.ORIENTATION_ROTATE_270 -> rotate(bmp, 270)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> rotate(bmp, 0, true)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> rotate(bmp, 180, true)
            ExifInterface.ORIENTATION_TRANSPOSE -> rotate(bmp, 90, true)
            ExifInterface.ORIENTATION_TRANSVERSE -> rotate(bmp, 270, true)
            else -> bmp
        }
        return scaled(up, 2048)
    }

    fun image(ctx: Context, uri: String): Map<String, Any?> {
        val bmp = decodeImage(ctx, uri)
        return mapOf("width" to bmp.width, "height" to bmp.height, "landmarks" to detect(singleDetector(), bmp))
    }

    fun video(ctx: Context, uri: String, fps: Double, maxDuration: Double, maxSize: Int, progress: (Double) -> Unit): Map<String, Any?> {
        val r = MediaMetadataRetriever()
        // Stream mode tracks the person from frame to frame: faster and steadier than detecting each frame alone.
        val detector = PoseDetection.getClient(
            PoseDetectorOptions.Builder().setDetectorMode(PoseDetectorOptions.STREAM_MODE).build())
        try {
            try {
                if (isContent(uri)) r.setDataSource(ctx, Uri.parse(uri)) else r.setDataSource(path(uri))
            } catch (e: Exception) {
                throw PoseDetectionError("cannot read video: ${uri.substringAfterLast('/')}")
            }
            if (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO) != "yes")
                throw PoseDetectionError("the file has no video track: ${uri.substringAfterLast('/')}")
            val durMs = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val vw = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val vh = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val rot = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            val length = minOf(durMs / 1000.0, maxDuration)
            val count = max(1, (length * fps).toInt())
            // Upright size: the frames come rotated already on most devices; rotate the ones that do not.
            val upW = if (rot % 180 != 0) vh else vw
            val upH = if (rot % 180 != 0) vw else vh
            val k = if (max(upW, upH) > maxSize) maxSize.toDouble() / max(upW, upH) else 1.0
            val outW = max(1, (upW * k).roundToInt())
            val outH = max(1, (upH * k).roundToInt())
            val frames = ArrayList<Map<String, Any?>>(count)
            for (i in 0 until count) {
                val t = i / fps
                val us = (t * 1_000_000).roundToLong()
                var bmp = if (Build.VERSION.SDK_INT >= 27) {
                    val sw = if (rot % 180 != 0) outH else outW
                    val sh = if (rot % 180 != 0) outW else outH
                    r.getScaledFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST, max(sw, sh), max(sw, sh))
                } else r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST)
                if (bmp != null) {
                    if (rot != 0 && (bmp.width > bmp.height) != (outW > outH) && outW != outH) bmp = rotate(bmp, rot)
                    if (bmp.width != outW || bmp.height != outH) bmp = Bitmap.createScaledBitmap(bmp, outW, outH, true)
                    frames.add(mapOf("t" to t, "landmarks" to detect(detector, bmp)))
                }
                progress((i + 1).toDouble() / count)
            }
            if (frames.isEmpty()) throw PoseDetectionError("could not read any frame of the video")
            return mapOf("width" to outW, "height" to outH, "duration" to length, "frames" to frames)
        } finally {
            detector.close()
            try { r.release() } catch (e: Exception) {}
        }
    }
}

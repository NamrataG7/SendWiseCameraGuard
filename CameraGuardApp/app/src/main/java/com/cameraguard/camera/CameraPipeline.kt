package com.cameraguard.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Handler
import android.os.Looper
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.cameraguard.media.AgeEstimationClassifier
import com.cameraguard.media.NsfwFrameClassifier
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.runBlocking
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.min

/** Snapshot of the result of processing a single frame, for UI/status updates. */
data class FrameState(
    val isProtected: Boolean,
    val totalFaceCount: Int,
    val minorFaceCount: Int
)

/**
 * Owns the CameraX pipeline: acquires frames via [ImageAnalysis] (no
 * [androidx.camera.view.PreviewView] — we need to bake blur into the pixels
 * before anything is ever displayed or saved), runs face detection + age
 * estimation per face and a throttled whole-frame NSFW check, applies blur
 * directly onto the frame bitmap on a background thread, then hands the
 * finished bitmap + state back to the caller on the main thread.
 *
 * Public API:
 *  - [faceBlurEnabled] / [nsfwBlurEnabled]: toggled live by the UI.
 *  - [start]: binds the analyzer to [lifecycleOwner] and begins processing.
 *  - [stop]: unbinds CameraX use cases and releases classifier resources.
 */
class CameraPipeline(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val onFrameReady: (Bitmap) -> Unit,
    private val onStateChanged: (FrameState) -> Unit
) {

    @Volatile var faceBlurEnabled: Boolean = true
    @Volatile var nsfwBlurEnabled: Boolean = true

    private val mainHandler = Handler(Looper.getMainLooper())
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    private val faceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()
    )
    private val ageClassifier = AgeEstimationClassifier(context)
    private val nsfwClassifier = NsfwFrameClassifier(context)

    // NSFW is a heavier whole-frame check; throttle it instead of running every frame.
    private val nsfwCheckIntervalMs = 500L
    @Volatile private var lastNsfwCheckAt = 0L
    @Volatile private var nsfwStickyFlag = false

    private val bannerPaint = Paint().apply {
        color = Color.argb(160, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val bannerTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        isAntiAlias = true
        textAlign = Paint.Align.CENTER
    }

    fun start() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy -> analyzeFrame(imageProxy) }

            provider.unbindAll()
            try {
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, imageAnalysis)
                cameraProvider = provider
            } catch (e: Exception) {
                // Fail open: no camera bound, nothing rendered; UI surfaces this via status text.
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        analysisExecutor.shutdown()
        faceDetector.close()
        ageClassifier.close()
        nsfwClassifier.close()
    }

    // ---- Per-frame pipeline -------------------------------------------------

    private fun analyzeFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            var minorFaceCount = 0
            var totalFaceCount = 0

            if (faceBlurEnabled) {
                val faces = detectFacesBlocking(bitmap)
                totalFaceCount = faces.size
                for (face in faces) {
                    val expanded = BlurUtils.expandRect(face, 0.2f, bitmap.width, bitmap.height)
                    if (expanded.width() < 2 || expanded.height() < 2) continue
                    val faceCrop = try {
                        Bitmap.createBitmap(bitmap, expanded.left, expanded.top, expanded.width(), expanded.height())
                    } catch (e: Exception) {
                        null
                    } ?: continue
                    val resized = Bitmap.createScaledBitmap(faceCrop, 224, 224, true)

                    val isMinor = try {
                        runBlocking { ageClassifier.estimateAge(resized).isMinor }
                    } catch (e: Exception) {
                        false // fail open
                    }
                    if (isMinor) {
                        minorFaceCount++
                        BlurUtils.blurRegion(bitmap, expanded)
                    }
                }
            }

            var isProtected = false
            if (nsfwBlurEnabled) {
                val now = System.currentTimeMillis()
                if (now - lastNsfwCheckAt >= nsfwCheckIntervalMs) {
                    lastNsfwCheckAt = now
                    nsfwStickyFlag = try {
                        val downscaled = downscaleForNsfwCheck(bitmap)
                        runBlocking { nsfwClassifier.classify(downscaled).isSensitive }
                    } catch (e: Exception) {
                        false // fail open
                    }
                }
                isProtected = nsfwStickyFlag
                if (isProtected) {
                    BlurUtils.boxBlur(bitmap, radius = 25, passes = 3)
                    drawWarningBanner(bitmap)
                }
            }

            val finalState = FrameState(isProtected, totalFaceCount, minorFaceCount)
            mainHandler.post {
                onFrameReady(bitmap)
                onStateChanged(finalState)
            }
        } catch (e: Exception) {
            // Never crash the camera pipeline because of a classifier/processing exception.
        } finally {
            imageProxy.close()
        }
    }

    private fun detectFacesBlocking(bitmap: Bitmap): List<Rect> {
        return try {
            val input = InputImage.fromBitmap(bitmap, 0)
            val faces = Tasks.await(faceDetector.process(input), 400, TimeUnit.MILLISECONDS)
            faces.map { it.boundingBox }
        } catch (e: Exception) {
            emptyList() // fail open
        }
    }

    private fun downscaleForNsfwCheck(bitmap: Bitmap): Bitmap {
        val targetWidth = 300
        if (bitmap.width <= targetWidth) return bitmap
        val scale = targetWidth.toFloat() / bitmap.width
        val targetHeight = (bitmap.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    private fun drawWarningBanner(bitmap: Bitmap) {
        val canvas = Canvas(bitmap)
        val bannerHeight = bitmap.height * 0.12f
        val top = (bitmap.height - bannerHeight) / 2f
        canvas.drawRect(0f, top, bitmap.width.toFloat(), top + bannerHeight, bannerPaint)
        bannerTextPaint.textSize = bannerHeight * 0.35f
        canvas.drawText(
            "Sensitive content detected — camera view protected",
            bitmap.width / 2f,
            top + bannerHeight / 2f + bannerTextPaint.textSize / 3f,
            bannerTextPaint
        )
    }

    // ---- YUV_420_888 -> Bitmap conversion (no RenderScript) -----------------

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val nv21 = yuv420ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bytes = out.toByteArray()
        var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

        val rotation = image.imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        }
        return bmp.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val nv21 = ByteArray(width * height * 3 / 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        var pos = 0
        if (yRowStride == width) {
            val ySize = yBuffer.remaining()
            yBuffer.get(nv21, 0, min(ySize, width * height))
            pos = width * height
        } else {
            val rowBuf = ByteArray(yRowStride)
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                val rowLen = min(yRowStride, yBuffer.remaining())
                yBuffer.get(rowBuf, 0, rowLen)
                System.arraycopy(rowBuf, 0, nv21, pos, width)
                pos += width
            }
        }

        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        val chromaHeight = height / 2
        val chromaWidth = width / 2

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val uIndex = row * uvRowStride + col * uvPixelStride
                val vIndex = row * vRowStride + col * vPixelStride
                // NV21 expects interleaved V,U after the Y plane.
                nv21[pos++] = safeGet(vBuffer, vIndex)
                nv21[pos++] = safeGet(uBuffer, uIndex)
            }
        }
        return nv21
    }

    private fun safeGet(buffer: java.nio.ByteBuffer, index: Int): Byte {
        return if (index in 0 until buffer.limit()) buffer.get(index) else 0
    }
}

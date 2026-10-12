package com.cameraguard.media

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * NsfwFrameClassifier - On-device NSFW/nudity classifier for live camera frames.
 *
 * Same bundled model as the SendWisePictureProtector keyboard app's
 * NsfwImageClassifier (GantMan/nsfw_model, MIT, MobileNetV2, 224x224x3 input,
 * 5-class softmax: drawings/hentai/neutral/porn/sexy) — reused here verbatim
 * since it is already proven mobile-appropriate (17MB, real-time capable).
 * See assets/models/MODEL_CARD_NSFW.md.
 *
 * FAIL-OPEN: any failure (missing model, bad input, inference error) returns
 * a non-sensitive "unavailable" result — this classifier never blocks/covers
 * the camera view due to an internal error.
 */
class NsfwFrameClassifier(private val context: Context) {

    companion object {
        private const val TAG = "NsfwFrameClassifier"
        private const val MODEL_ASSET_PATH = "models/nsfw_classifier.tflite"
        private const val INPUT_IMAGE_SIZE = 224
        private const val INPUT_CHANNELS = 3
        private const val PIXEL_MAX_VALUE = 255.0f

        private const val LABEL_DRAWINGS = "drawings"
        private const val LABEL_HENTAI = "hentai"
        private const val LABEL_NEUTRAL = "neutral"
        private const val LABEL_PORN = "porn"
        private const val LABEL_SEXY = "sexy"
        private const val LABEL_UNAVAILABLE = "unavailable"

        val LABELS: List<String> = listOf(LABEL_DRAWINGS, LABEL_HENTAI, LABEL_NEUTRAL, LABEL_PORN, LABEL_SEXY)

        const val DEFAULT_SENSITIVE_THRESHOLD = 0.5f
    }

    data class Result(
        val isSensitive: Boolean,
        val confidence: Float,
        val label: String
    )

    private val interpreter: Interpreter? by lazy {
        try {
            Interpreter(loadModelFile(MODEL_ASSET_PATH))
        } catch (t: Throwable) {
            Log.w(TAG, "NSFW frame model failed to load: ${t.message}")
            null
        }
    }

    suspend fun classify(
        bitmap: Bitmap,
        sensitiveThreshold: Float = DEFAULT_SENSITIVE_THRESHOLD
    ): Result = withContext(Dispatchers.Default) {
        try {
            val clf = interpreter ?: run {
                Log.w(TAG, "NSFW frame classifier unavailable — failing open")
                return@withContext unavailableResult()
            }
            val input = preprocess(bitmap)
            val output = Array(1) { FloatArray(LABELS.size) }
            clf.run(input, output)
            postprocess(output[0], sensitiveThreshold)
        } catch (t: Throwable) {
            Log.w(TAG, "NSFW frame classification failed, failing open: ${t.message}", t)
            unavailableResult()
        }
    }

    fun close() {
        try {
            interpreter?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "Error closing NSFW frame interpreter: ${t.message}")
        }
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = if (bitmap.width == INPUT_IMAGE_SIZE && bitmap.height == INPUT_IMAGE_SIZE) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true)
        }

        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * INPUT_CHANNELS)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE)
        resized.getPixels(pixels, 0, INPUT_IMAGE_SIZE, 0, 0, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            byteBuffer.putFloat(r / PIXEL_MAX_VALUE)
            byteBuffer.putFloat(g / PIXEL_MAX_VALUE)
            byteBuffer.putFloat(b / PIXEL_MAX_VALUE)
        }

        if (resized !== bitmap) resized.recycle()
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun postprocess(probabilities: FloatArray, sensitiveThreshold: Float): Result {
        val hentaiIdx = LABELS.indexOf(LABEL_HENTAI)
        val pornIdx = LABELS.indexOf(LABEL_PORN)
        val sexyIdx = LABELS.indexOf(LABEL_SEXY)

        val sensitiveMass = (probabilities.getOrNull(hentaiIdx) ?: 0f) +
            (probabilities.getOrNull(pornIdx) ?: 0f) +
            (probabilities.getOrNull(sexyIdx) ?: 0f)

        var topIdx = 0
        for (i in probabilities.indices) {
            if (probabilities[i] > probabilities[topIdx]) topIdx = i
        }
        val topLabel = LABELS.getOrElse(topIdx) { LABEL_NEUTRAL }
        val topConfidence = probabilities.getOrElse(topIdx) { 0f }

        return Result(
            isSensitive = sensitiveMass >= sensitiveThreshold,
            confidence = topConfidence,
            label = topLabel
        )
    }

    private fun unavailableResult(): Result = Result(isSensitive = false, confidence = 0f, label = LABEL_UNAVAILABLE)

    private fun loadModelFile(assetPath: String): MappedByteBuffer {
        context.assets.openFd(assetPath).use { afd ->
            FileInputStream(afd.fileDescriptor).use { input ->
                val channel = input.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }
}

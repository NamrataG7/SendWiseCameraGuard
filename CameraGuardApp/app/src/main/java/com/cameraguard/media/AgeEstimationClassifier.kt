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
 * AgeEstimationClassifier - On-device face-age estimator used to decide
 * whether a detected face belongs to a minor and should be blurred.
 *
 * STATUS: integration-complete, model NOT bundled (see
 * assets/models/MODEL_CARD_AGE.md for the full investigation). In short:
 * the one MIT-licensed, commercially-bundlable candidate found during
 * research (serengil/deepface's VGG-Face-backbone age model) was actually
 * converted and benchmarked here — it produced a 443MB float32 / 135MB
 * int8-quantized `.tflite` file. Both exceed GitHub's 100MB single-file
 * push limit, and a VGG16-class backbone is too slow for real-time
 * per-face inference on mobile hardware regardless of hosting. Rather than
 * bundle a broken/impractical artifact, this class is wired end-to-end and
 * ready to load any correctly-shaped replacement model — it just fails
 * open (never flags a face as a minor) until one is added.
 *
 * Expected model contract (see MODEL_CARD_AGE.md for a recommended
 * lightweight MobileNetV2-based replacement path):
 *   - Input: [1, 224, 224, 3] float32, RGB, normalized [0, 1]
 *   - Output: [1, 101] float32 softmax over ages 0..100 (DEX-style
 *     "expected value" convention: estimated age = sum(i * p_i))
 *
 * FAIL-OPEN: any failure (missing model, bad input, inference error)
 * returns isMinor = false — this classifier never blurs/blocks due to an
 * internal error, consistent with every other classifier in this project.
 */
class AgeEstimationClassifier(private val context: Context) {

    companion object {
        private const val TAG = "AgeEstimationClassifier"
        private const val MODEL_ASSET_PATH = "models/age_classifier.tflite"
        private const val INPUT_IMAGE_SIZE = 224
        private const val INPUT_CHANNELS = 3
        private const val PIXEL_MAX_VALUE = 255.0f
        private const val NUM_AGE_CLASSES = 101

        const val DEFAULT_MINOR_THRESHOLD = 21f
    }

    data class AgeResult(val estimatedAge: Float, val isMinor: Boolean)

    private val interpreter: Interpreter? by lazy {
        try {
            Interpreter(loadModelFile(MODEL_ASSET_PATH))
        } catch (t: Throwable) {
            Log.w(TAG, "Age model not available (${t.message}) — age-based face blur is a no-op until a model is added. See MODEL_CARD_AGE.md.")
            null
        }
    }

    suspend fun estimateAge(
        faceBitmap: Bitmap,
        minorThreshold: Float = DEFAULT_MINOR_THRESHOLD
    ): AgeResult = withContext(Dispatchers.Default) {
        try {
            val clf = interpreter ?: return@withContext unavailableResult()
            val input = preprocess(faceBitmap)
            val output = Array(1) { FloatArray(NUM_AGE_CLASSES) }
            clf.run(input, output)

            val probabilities = output[0]
            var expectedAge = 0f
            for (i in probabilities.indices) {
                expectedAge += i * probabilities[i]
            }
            AgeResult(estimatedAge = expectedAge, isMinor = expectedAge < minorThreshold)
        } catch (t: Throwable) {
            Log.w(TAG, "Age estimation failed, failing open: ${t.message}", t)
            unavailableResult()
        }
    }

    fun close() {
        try {
            interpreter?.close()
        } catch (t: Throwable) {
            Log.w(TAG, "Error closing age interpreter: ${t.message}")
        }
    }

    private fun unavailableResult(): AgeResult = AgeResult(estimatedAge = -1f, isMinor = false)

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

    private fun loadModelFile(assetPath: String): MappedByteBuffer {
        context.assets.openFd(assetPath).use { afd ->
            FileInputStream(afd.fileDescriptor).use { input ->
                val channel = input.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }
}

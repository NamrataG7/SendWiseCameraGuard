package com.cameraguard.camera

import android.graphics.Bitmap
import android.graphics.Rect
import kotlin.math.max
import kotlin.math.min

/**
 * RenderScript-free box-blur helpers. RenderScript is deprecated on recent Android
 * versions, so we use a classic multi-pass horizontal+vertical box blur (the same
 * building block as a "stack blur") operating directly on an ARGB pixel array.
 *
 * These functions mutate the pixel contents of the given [bitmap] in place and
 * return it for call-site chaining convenience. [bitmap] must be mutable
 * (e.g. created via [Bitmap.copy] with `isMutable = true`, or already mutable
 * as produced by [android.graphics.BitmapFactory] decoding).
 */
object BlurUtils {

    /** Blurs the entire [bitmap] with the given [radius], repeated [passes] times. */
    fun boxBlur(bitmap: Bitmap, radius: Int = 18, passes: Int = 3): Bitmap {
        if (radius <= 0 || bitmap.width < 2 || bitmap.height < 2) return bitmap
        val region = Rect(0, 0, bitmap.width, bitmap.height)
        blurRegion(bitmap, region, radius, passes)
        return bitmap
    }

    /**
     * Blurs only the sub-rectangle [rect] of [bitmap] (clamped to bitmap bounds),
     * leaving the rest of the bitmap untouched. Useful for blurring just a
     * detected face region without paying the cost of blurring the whole frame.
     */
    fun blurRegion(bitmap: Bitmap, rect: Rect, radius: Int = 14, passes: Int = 2): Bitmap {
        if (radius <= 0) return bitmap
        val clamped = Rect(
            max(0, rect.left),
            max(0, rect.top),
            min(bitmap.width, rect.right),
            min(bitmap.height, rect.bottom)
        )
        val w = clamped.width()
        val h = clamped.height()
        if (w < 2 || h < 2) return bitmap

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, clamped.left, clamped.top, w, h)

        repeat(passes) {
            horizontalPass(pixels, w, h, radius)
            verticalPass(pixels, w, h, radius)
        }

        bitmap.setPixels(pixels, 0, w, clamped.left, clamped.top, w, h)
        return bitmap
    }

    private fun horizontalPass(pixels: IntArray, w: Int, h: Int, radius: Int) {
        val row = IntArray(w)
        for (y in 0 until h) {
            val rowOffset = y * w
            System.arraycopy(pixels, rowOffset, row, 0, w)
            for (x in 0 until w) {
                var a = 0; var r = 0; var g = 0; var b = 0; var count = 0
                val from = max(0, x - radius)
                val to = min(w - 1, x + radius)
                for (nx in from..to) {
                    val p = row[nx]
                    a += (p ushr 24) and 0xFF
                    r += (p ushr 16) and 0xFF
                    g += (p ushr 8) and 0xFF
                    b += p and 0xFF
                    count++
                }
                pixels[rowOffset + x] = averagedPixel(a, r, g, b, count)
            }
        }
    }

    private fun verticalPass(pixels: IntArray, w: Int, h: Int, radius: Int) {
        val col = IntArray(h)
        for (x in 0 until w) {
            for (y in 0 until h) col[y] = pixels[y * w + x]
            for (y in 0 until h) {
                var a = 0; var r = 0; var g = 0; var b = 0; var count = 0
                val from = max(0, y - radius)
                val to = min(h - 1, y + radius)
                for (ny in from..to) {
                    val p = col[ny]
                    a += (p ushr 24) and 0xFF
                    r += (p ushr 16) and 0xFF
                    g += (p ushr 8) and 0xFF
                    b += p and 0xFF
                    count++
                }
                pixels[y * w + x] = averagedPixel(a, r, g, b, count)
            }
        }
    }

    private fun averagedPixel(a: Int, r: Int, g: Int, b: Int, count: Int): Int {
        return ((a / count) and 0xFF shl 24) or
            ((r / count) and 0xFF shl 16) or
            ((g / count) and 0xFF shl 8) or
            ((b / count) and 0xFF)
    }

    /** Expands [rect] by [marginFraction] on each side, clamped to [maxWidth]/[maxHeight]. */
    fun expandRect(rect: Rect, marginFraction: Float, maxWidth: Int, maxHeight: Int): Rect {
        val marginX = (rect.width() * marginFraction).toInt()
        val marginY = (rect.height() * marginFraction).toInt()
        return Rect(
            max(0, rect.left - marginX),
            max(0, rect.top - marginY),
            min(maxWidth, rect.right + marginX),
            min(maxHeight, rect.bottom + marginY)
        )
    }
}

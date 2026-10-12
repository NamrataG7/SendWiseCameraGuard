package com.cameraguard.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Plain [View] that draws the latest processed (already blur-applied) camera
 * frame bitmap. We intentionally avoid CameraX's [androidx.camera.view.PreviewView]
 * here because it renders the raw camera feed directly to a Surface and offers
 * no hook to bake per-region or full-frame blur into what's actually displayed.
 *
 * [updateFrame] must be called from the main thread; the analyzer pipeline is
 * responsible for posting the finished bitmap here after processing it on a
 * background thread.
 */
class ProcessedCameraView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var frame: Bitmap? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private val matrix = Matrix()

    /** Replaces the currently rendered frame and triggers a redraw. */
    fun updateFrame(bitmap: Bitmap) {
        frame = bitmap
        invalidate()
    }

    /** Returns the frame currently on screen, for use by the capture flow. */
    fun currentFrame(): Bitmap? = frame

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bmp = frame ?: return

        // Center-crop/scale the source bitmap to fill the view bounds while
        // preserving aspect ratio, mirroring typical camera preview behavior.
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val bmpW = bmp.width.toFloat()
        val bmpH = bmp.height.toFloat()
        if (viewW <= 0f || viewH <= 0f || bmpW <= 0f || bmpH <= 0f) return

        val scale = max(viewW / bmpW, viewH / bmpH)
        val scaledW = bmpW * scale
        val scaledH = bmpH * scale
        val dx = (viewW - scaledW) / 2f
        val dy = (viewH - scaledH) / 2f

        matrix.reset()
        matrix.postScale(scale, scale)
        matrix.postTranslate(dx, dy)

        canvas.drawBitmap(bmp, matrix, paint)
    }

    private fun max(a: Float, b: Float) = if (a > b) a else b
}

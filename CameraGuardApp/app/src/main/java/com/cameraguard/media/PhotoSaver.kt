package com.cameraguard.media

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Saves bitmaps to the device's shared Pictures collection via MediaStore
 * (scoped storage, API 29+ friendly — no WRITE_EXTERNAL_STORAGE permission needed).
 *
 * IMPORTANT: Callers must only ever pass the already-processed (blur-applied)
 * bitmap here. There is intentionally no API for saving a raw/unprocessed frame —
 * this class has no knowledge of "original" frames at all.
 */
object PhotoSaver {

    private const val DISPLAY_NAME_PREFIX = "CameraGuard_"
    private const val MIME_TYPE = "image/jpeg"
    private const val RELATIVE_DIR = "Pictures/CameraGuard"

    /**
     * Persists [processedBitmap] as a new JPEG in MediaStore.Images.
     * Returns the resulting content Uri, or null if the save failed.
     */
    fun saveProcessedPhoto(context: Context, processedBitmap: Bitmap): android.net.Uri? {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "$DISPLAY_NAME_PREFIX$timestamp.jpg"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, MIME_TYPE)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, RELATIVE_DIR)
                put(MediaStore.Images.Media.IS_PENDING, 1)
            } else {
                @Suppress("DEPRECATION")
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                put(MediaStore.Images.Media.DATA, "${dir.absolutePath}/$fileName")
            }
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null

        return try {
            resolver.openOutputStream(uri)?.use { out: OutputStream ->
                processedBitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            uri
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            null
        }
    }
}

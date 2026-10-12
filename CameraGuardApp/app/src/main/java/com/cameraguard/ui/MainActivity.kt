package com.cameraguard.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.cameraguard.R
import com.cameraguard.camera.CameraPipeline
import com.cameraguard.camera.FrameState
import com.cameraguard.databinding.ActivityMainBinding
import com.cameraguard.media.PhotoSaver
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var cameraPipeline: CameraPipeline? = null

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            onPermissionGranted()
        } else {
            binding.permissionContainer.visibility = android.view.View.VISIBLE
            Toast.makeText(this, R.string.permission_denied_toast, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.grantPermissionButton.setOnClickListener {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }

        binding.faceBlurToggle.setOnCheckedChangeListener { _, isChecked ->
            cameraPipeline?.faceBlurEnabled = isChecked
        }
        binding.nsfwBlurToggle.setOnCheckedChangeListener { _, isChecked ->
            cameraPipeline?.nsfwBlurEnabled = isChecked
        }

        binding.captureButton.setOnClickListener { capturePhoto() }

        if (hasCameraPermission()) {
            onPermissionGranted()
        } else {
            binding.permissionContainer.visibility = android.view.View.VISIBLE
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun onPermissionGranted() {
        binding.permissionContainer.visibility = android.view.View.GONE
        startPipeline()
    }

    private fun startPipeline() {
        if (cameraPipeline != null) return

        val pipeline = CameraPipeline(
            context = this,
            lifecycleOwner = this,
            onFrameReady = { bitmap -> binding.processedCameraView.updateFrame(bitmap) },
            onStateChanged = { state -> updateStatusBanner(state) }
        )
        pipeline.faceBlurEnabled = binding.faceBlurToggle.isChecked
        pipeline.nsfwBlurEnabled = binding.nsfwBlurToggle.isChecked
        pipeline.start()
        cameraPipeline = pipeline
    }

    private fun updateStatusBanner(state: FrameState) {
        binding.statusBanner.text = when {
            state.isProtected -> getString(R.string.status_protected)
            state.totalFaceCount > 0 -> getString(R.string.status_scanning)
            else -> getString(R.string.status_clear)
        }
    }

    private fun capturePhoto() {
        // Only ever persist the currently rendered, already-processed (blur-applied)
        // bitmap. There is no path here that touches a raw/unprocessed frame.
        val processed = binding.processedCameraView.currentFrame()
        if (processed == null) {
            Toast.makeText(this, R.string.photo_save_failed, Toast.LENGTH_SHORT).show()
            return
        }
        val snapshot = processed.copy(processed.config ?: android.graphics.Bitmap.Config.ARGB_8888, false)
        CoroutineScope(Dispatchers.IO).launch {
            val uri = PhotoSaver.saveProcessedPhoto(applicationContext, snapshot)
            launch(Dispatchers.Main) {
                val messageRes = if (uri != null) R.string.photo_saved else R.string.photo_save_failed
                Toast.makeText(this@MainActivity, messageRes, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        cameraPipeline?.stop()
        cameraPipeline = null
        super.onDestroy()
    }
}

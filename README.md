# SendWise CameraGuard

**On-device camera: real-time nudity blur + minor-face blur, for parental safety.**

Companion app to [SendWisePictureProtector](https://github.com/NamrataG7/SendWisePictureProtector) (the keyboard app) — same privacy model: **everything runs on-device, nothing is uploaded, nothing unblurred is ever persisted.**

This is a **parental safety tool** — installed with a parent's knowledge and consent on a device they own/manage. It does **not** attempt to detect or report CSAM. It helps with two specific risks:

1. **Live nudity detection** — the camera preview is continuously scanned on-device; if sensitive content is detected, the entire view is blurred with a visible warning until it clears.
2. **Minor-face blur** — faces are detected via Google ML Kit (on-device, no cloud calls, no identity recognition), and faces estimated to belong to a minor are blurred in real time to protect their identity before any photo is taken.

Captured photos always reflect the already-blurred, protected view — there is no code path that saves a raw/unblurred frame.

---

## Architecture

- **Face detection**: Google ML Kit Face Detection (`com.google.mlkit:face-detection`), on-device, bounding boxes only — ML Kit does not perform identity recognition.
- **Age estimation**: a TFLite classifier estimates each detected face's age; faces below a configurable threshold are blurred. **See "Current status" below — this model is not yet bundled.**
- **NSFW/nudity detection**: the same bundled, MIT-licensed `nsfw_classifier.tflite` model used by the SendWisePictureProtector keyboard app (GantMan/nsfw_model, MobileNetV2, 17MB) — reused verbatim here, run periodically (not every frame) on a downscaled copy of the live view.
- **Rendering**: CameraX `ImageAnalysis` (not `PreviewView`) feeds raw frames to a custom pipeline that converts YUV→Bitmap, applies blur directly to the pixels of regions that need it, and renders the processed bitmap to a custom view. This means the blur is baked into what's actually displayed and captured — not an overlay that could be bypassed.
- **Capture**: `MediaStore` (scoped storage, no legacy permission), saving only the already-processed (blur-applied) frame.

## Current build/verification status

| Component | Status |
|---|---|
| CameraX pipeline, face detection, blur rendering, capture/save flow | ✅ Implemented — see `CameraGuardApp/app/src/main/java/com/cameraguard/` |
| NSFW/nudity frame classifier | ✅ Real, bundled, same model proven in the keyboard app |
| **Age estimation classifier** | ⚠️ **Integration complete, model NOT bundled.** A real candidate (serengil/deepface's MIT-licensed age model) was sourced, converted to TFLite, and benchmarked — it came out to 443MB (float32) / 135MB (int8-quantized), both impractical: over GitHub's 100MB file limit, and a VGG16-class backbone is too slow for real-time multi-face mobile inference regardless of hosting. See `CameraGuardApp/app/src/main/assets/models/MODEL_CARD_AGE.md` for the full investigation and what a suitable replacement model needs to look like. **Until a lightweight replacement is added, `AgeEstimationClassifier` fails open — faces are detected but never flagged as minors, so face blur is currently a no-op.** |
| Android build (Gradle/CI) | ⏳ Pushed, CI triggered — check the [Actions tab](https://github.com/NamrataG7/SendWiseCameraGuard/actions) for the latest run result |
| On-device runtime behavior | ❌ Not yet tested on an emulator/device |
| Legal/compliance review | ❌ Not done — see the same note in SendWisePictureProtector's README; applies here too |

## Not yet verified (compiles ≠ correct at runtime)

- No emulator/physical device run has happened — install the CI-built debug APK and manually verify: camera preview renders, NSFW blur triggers on a test image/scene, capture saves the (blurred) MediaStore photo correctly, permission flow works.
- Real-time performance (frames-per-second with face detection + periodic NSFW classification running) has not been profiled on real hardware.
- The 500ms NSFW-check throttle and the 20% face-crop margin are reasonable starting defaults, not tuned against real footage.

## License

MIT — see `LICENSE`. Bundles the MIT-licensed `nsfw_classifier.tflite` from GantMan/nsfw_model (`THIRD_PARTY_LICENSES/nsfw_model-LICENSE.md`).

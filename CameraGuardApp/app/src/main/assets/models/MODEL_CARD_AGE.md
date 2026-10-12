# Model Card — Face Age Estimation

## Status: integration complete, model NOT bundled (by deliberate decision)

This is not a "we didn't get to it" gap — a real candidate model was sourced,
converted, and benchmarked, then rejected for concrete, measured reasons
documented below. `AgeEstimationClassifier` fails open (never flags a face
as a minor) until a suitable replacement is added at
`app/src/main/assets/models/age_classifier.tflite`.

## What was tried

**Candidate: [serengil/deepface](https://github.com/serengil/deepface)'s Age
model** (`age_model_weights.h5`, MIT license via
[serengil/deepface_models](https://github.com/serengil/deepface_models),
VGG-Face backbone + a 101-class DEX-style "expected value" age head).

This was the only candidate found during research with a clean,
commercially-bundlable license (see the research summary below) — other
options were rejected on licensing grounds before any conversion was
attempted:

- `yu4u/age-gender-estimation` (UTKFace weights): code is MIT, but the
  README explicitly restricts the pretrained weights to **academic
  research use only** (IMDB-WIKI dataset terms) — not usable here.
- Levi/Hassner "Adience" Caffe age model (`age_net.caffemodel` /
  `deploy_age.prototxt`, referenced in this project's original Python
  prototypes): original dataset/model license is **non-commercial research
  use only**, the lineage is unmaintained (Caffe-era, ~2015), and no
  TFLite port exists.
- MediaPipe does **not** ship any age-estimation model at all — confirmed
  directly against Google's MediaPipe Face Detector/Face Landmarker docs
  (bounding boxes / landmarks / blendshapes only, no demographic head).

### Conversion result — why it was rejected

The VGG-Face backbone (base model shared with deepface's face-recognition
pipeline) was reconstructed in Keras, had both the `vgg_face_weights.h5`
(580 MB) and `age_model_weights.h5` (539 MB) loaded onto it, and was
converted to TensorFlow Lite:

| Conversion | Size | Verdict |
|---|---|---|
| Float32 (no quantization) | **443 MB** | Rejected — far too large for a mobile app and for a git-committed asset |
| Int8 (full integer quantization, `tf.lite.Optimize.DEFAULT` + representative dataset) | **135 MB** | Rejected — still exceeds GitHub's 100 MB single-file push limit, **and** a VGG16-class backbone (the `Convolution2D(4096,(7,7))` / `Convolution2D(4096,(1,1))` layers alone account for the bulk of the ~118M parameters) is too slow for real-time per-face inference on mobile CPUs regardless of where the file is hosted |

In short: this is **not a file-hosting problem that Git LFS or a GitHub
Release asset would solve** — the architecture itself is the wrong choice
for a real-time, multiple-faces-per-frame mobile pipeline. Bundling it
(even externally-hosted) would have produced an app that technically "has"
age estimation but is too slow to actually run on live video, which would
have been a misleading deliverable.

## What's needed to complete this feature

A **MobileNet-class (or smaller) age-regression/classification model**,
ideally:
- Trained on UTKFace or a similarly permissively-licensed dataset, with
  **both the code AND the pretrained weights** under a commercially-usable
  license (MIT/Apache/BSD) — verify this explicitly; many public
  "age-estimation" GitHub repos have MIT *code* but research-only *weights*
  trained on IMDB-WIKI, which looks permissive at a glance but is not.
- Input around 64×64 to 224×224, single regression output (age in years)
  or a small number of age-bucket classes — avoid 101-way softmax +
  VGG16-depth backbones.
- Target: under ~10 MB `.tflite`, sub-50ms inference per face crop on a
  mid-range phone CPU, to support multiple faces per frame in real time.

If/when such a model is found or trained, convert it to TFLite, place it
at `app/src/main/assets/models/age_classifier.tflite`, and verify its
output convention matches what `AgeEstimationClassifier` expects — or
adjust the postprocessing in that class to match the new model's output
format (e.g. if the replacement outputs a single raw regression float
instead of a 101-class softmax, replace the "expected value" computation
with a direct read of that value).

## Expected tensor contract (for any replacement model)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| Input | `[1, 224, 224, 3]` | `float32` | NHWC, RGB, normalized to `[0, 1]` |
| Output | `[1, 101]` | `float32` | Softmax over ages 0–100; estimated age = Σ(i × pᵢ) |

(Adjust `AgeEstimationClassifier.preprocess()`/`estimateAge()` if a
replacement model uses a different input size or a direct regression
output instead.)

## Non-negotiables (mirrors every other classifier in this project)

- No cloud inference — must run entirely on-device.
- No image upload, no logging of face crops/bytes.
- Fail-open: if the model is missing/fails, never flag a face as a minor
  rather than risk blocking/crashing on a false positive from a broken
  classifier.

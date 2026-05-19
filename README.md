# face-pose-tracker

Realtime face mesh + head pose + ARKit-style blendshape streamer for game
characters. Webcam in, **478 3D landmarks + 4×4 head transformation matrix +
52 facial expression coefficients** out, ~30 Hz, over a local TCP socket so
any engine (Unity, Unreal, Godot, Blender, custom) can drive a character
with the player's face.

The goal: a developer-side "FaceID" you can drop into an open test where any
visitor sits in front of a webcam, becomes a character in your game, and the
game applies *its own* textures and rig.

There is also a **bundled `.exe`** so non-developers can run it without
touching Python — see [Building the .exe](#building-the-exe).

---

## Table of contents

- [Install / run](#install--run)
- [Stream protocol](#stream-protocol)
- [How the 478 points are computed](#how-the-478-points-are-computed)
- [Coordinate systems & normalization](#coordinate-systems--normalization)
- [Landmark index reference](#landmark-index-reference)
- [Head pose: the 4×4 transformation matrix](#head-pose-the-4x4-transformation-matrix)
- [Blendshapes (52 ARKit-style)](#blendshapes-52-arkit-style)
- [Face oval, contours, tesselation, irises](#face-oval-contours-tesselation-irises)
- [UV unwrap & static asset export](#uv-unwrap--static-asset-export)
- [Game engine integration](#game-engine-integration)
- [Building the .exe](#building-the-exe)
- [What to rely on, what NOT to rely on](#what-to-rely-on-what-not-to-rely-on)
- [Performance notes](#performance-notes)

---

## Install / run

```bash
pip install -r requirements.txt
python detect.py
```

Python 3.10 – 3.12. On Windows enable camera access in
Privacy settings → Camera → *Let desktop apps access your camera*.

On first run the script auto-downloads `face_landmarker.task` (~3.7 MB) next
to itself (or next to the `.exe` if frozen). If you're offline, grab it
manually from
[storage.googleapis.com/mediapipe-models/...](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
and put it beside `detect.py`.

Verify the live stream from a second terminal:

```bash
python client_example.py
```

### Keys

| key      | action |
|----------|---------------------------------------------------------------|
| `1`–`4`  | render mode: wireframe / textured / textured+wire / dense    |
| `i`      | toggle iris ring rendering                                    |
| `c`      | toggle contour rendering                                      |
| `p`      | toggle head pose axes overlay                                 |
| `r`      | toggle JSONL stream recording → `./recordings/rec_TS.jsonl`   |
| `v`      | toggle MP4 video of the canvas → `./videos/video_TS.mp4`      |
| `s`      | export static asset bundle → `./exports/face_TS.{obj,mtl,_uv.png,_cutout.png}` |
| `Esc`/`q`| quit                                                          |

---

## Stream protocol

TCP, listens on `127.0.0.1:8765`, **one JSON object per line**, ~30 Hz when a
face is in frame.

When a face is detected:

```jsonc
{
  "ts": 1736012345.123,                       // unix time, float
  "ok": true,
  "landmarks": [[x,y,z], ... 478 entries],    // see "Coordinate systems"
  "head_matrix": [[r00,r01,r02,tx],           // 4x4 row-major
                  [r10,r11,r12,ty],
                  [r20,r21,r22,tz],
                  [ 0,  0,  0,  1]],
  "head_euler": [pitch, yaw, roll],           // degrees, derived from head_matrix
  "blendshapes": {                            // 52 ARKit-equivalent keys, 0..1
    "_neutral": 0.0,
    "jawOpen": 0.13,
    "eyeBlinkLeft": 0.04,
    "mouthSmileRight": 0.21,
    "browInnerUp": 0.0,
    ... (full list below)
  }
}
```

When no face is in frame (tracking lost):

```json
{ "ts": 1736012345.456, "ok": false }
```

This lets the game show a "tracking lost" UI instead of guessing.

---

## How the 478 points are computed

The pipeline inside MediaPipe FaceLandmarker is:

```
 webcam frame (any size)
        │
        ▼
 ┌──────────────────────┐
 │ face detector (SSD)  │  ── BlazeFace, runs every frame in detection-only
 └──────────┬───────────┘     mode or periodically in tracking mode; localizes
            │                 a square ROI around the face
            ▼
 ┌──────────────────────┐
 │ face landmark net    │  ── input: 256×256 RGB crop of the ROI
 │ (attention mesh)     │     architecture: an attention-mesh CNN; outputs
 └──────────┬───────────┘     468 3D landmark offsets + iris/lip refinements
            │
            ▼
 ┌──────────────────────┐
 │ blendshape head      │  ── input: subset of refined landmarks
 │ (small MLP)          │     output: 52 expression coefficients ∈ [0,1]
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ canonical-model fit  │  ── solves a rigid transform from the
 │ (Procrustes-ish)     │     canonical face model to the detected
 └──────────┬───────────┘     landmarks → produces head_matrix
            │
            ▼
 ┌──────────────────────┐
 │ output               │  ── 478 NormalizedLandmark + 52 Category +
 └──────────────────────┘     facial_transformation_matrix
```

Key facts:

- **478 = 468 face mesh + 10 iris** (5 per eye: 1 center + 4 ring).
- The original face model is the canonical [MediaPipe "attention mesh"](https://arxiv.org/abs/2006.10962); the iris and blendshape heads were added later (the model file is `float16/1/face_landmarker.task`).
- The model is trained on a synthetic + manually annotated face dataset that covers a wide range of head poses (±90° yaw before tracking breaks down), expressions, lighting, ethnicity, and partial occlusion (e.g. one hand on cheek).
- The script runs the model in **`RunningMode.VIDEO`** with **XNNPACK CPU acceleration** by default. To switch to GPU delegate (CUDA / DirectML where supported), change `BaseOptions(delegate=...)` in `create_detector()`.
- One face at a time (`num_faces=1`). Multi-tracking works — bump the constant and iterate `result.face_landmarks`.

---

## Coordinate systems & normalization

| field         | system                                                                 |
|---------------|------------------------------------------------------------------------|
| `landmarks.x` | normalized image space, `0.0 = left edge`, `1.0 = right edge`          |
| `landmarks.y` | normalized image space, `0.0 = top edge`, `1.0 = bottom edge`          |
| `landmarks.z` | roughly metric depth, scaled so that face *width* ≈ 1.0 in z-units; **negative z** means closer to the camera than the face center; **positive z** means farther (e.g. the back of the head plane) |
| `head_matrix` | 4×4 transform from the canonical face model (centered at face origin, +Y up, +X to the subject's left, +Z forward toward the camera) to a world frame whose units are roughly centimeters with the camera at the origin |

> **Important:** the camera frame is mirrored (`cv2.flip(frame, 1)`) before
> inference so that "screen-left" matches "subject's own right hand," which is
> the convention used by the rest of this README and by MediaPipe's own
> `FACEMESH_LEFT_*` constants. If you want raw-camera coordinates instead, remove
> the `cv2.flip` in `run()`.

To convert a landmark to **pixel coordinates** in a frame of size `W × H`:

```python
px, py = lm.x * W, lm.y * H
```

To convert a landmark to **canonical face-local coords** (e.g. for fitting a
mesh in Blender / a game engine), apply the inverse of `head_matrix`.

---

## Landmark index reference

These are the indices you'll actually reach for. They match MediaPipe's
canonical numbering — same indices, same vertices across captures, across
people, and across versions.

### Single anchor points

| index | location                          |
|-------|-----------------------------------|
| 1     | nose tip                          |
| 4     | nose bridge (between nostrils)    |
| 6     | nose bridge top (between eyes)    |
| 9     | between eyebrows (glabella)       |
| 10    | center of forehead, hairline      |
| 13    | upper lip center, inner edge      |
| 14    | lower lip center, inner edge      |
| 17    | lower lip center, outer edge      |
| 0     | upper lip center, outer edge      |
| 152   | chin tip                          |
| 168   | nasal bridge top (just below 9)   |
| 175   | chin lower point                  |
| 199   | jaw center under chin             |
| 234   | left ear connection (cheek edge)  |
| 454   | right ear connection (cheek edge) |

### Eyes (screen-left = subject's right)

| feature          | left-eye index | right-eye index |
|------------------|----------------|-----------------|
| outer corner     | 33             | 263             |
| inner corner     | 133            | 362             |
| upper lid top    | 159            | 386             |
| lower lid bottom | 145            | 374             |

The **EAR (eye aspect ratio)** sextet used in the original blink detector:

```
left  = [33, 160, 158, 133, 153, 144]
right = [362, 385, 387, 263, 373, 380]
```

### Irises (only with `refine_landmarks=True`, which `FaceLandmarker` enables)

| feature          | left | right |
|------------------|------|-------|
| iris center      | 468  | 473   |
| iris ring (4 pts)| 469–472 | 474–477 |

For gaze, compare iris center position against the eye corners:

```
gaze_x = ((iris_center − midpoint(corners)) · (inner − outer))
       / |inner − outer|²
```

### Lips

| feature           | index |
|-------------------|-------|
| left mouth corner | 61    |
| right mouth corner| 291   |
| upper lip top     | 0     |
| upper lip inner   | 13    |
| lower lip inner   | 14    |
| lower lip bottom  | 17    |

### Eyebrows

| feature        | left  | right |
|----------------|-------|-------|
| outer end      | 70    | 300   |
| middle         | 105   | 334   |
| inner end      | 107   | 336   |

### Full feature loops

These are imported live from MediaPipe — don't hard-code them, ask the lib:

```python
from mediapipe.solutions.face_mesh import (
    FACEMESH_FACE_OVAL,      # 36 edges, closed loop, outer silhouette
    FACEMESH_LIPS,           # outer + inner lip ring
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_CONTOURS,       # union of all of the above
    FACEMESH_TESSELATION,    # ~2556 edges → 854 triangles (see below)
)
```

---

## Head pose: the 4×4 transformation matrix

`facial_transformation_matrixes[0]` is a row-major 4×4 that maps a point in
the **canonical face model** (mediapipe's reference average face, in cm,
centered at the face origin) into world space (camera-relative, cm):

```
┌                                ┐
│ r00  r01  r02  tx │  rotation  │
│ r10  r11  r12  ty │     +      │
│ r20  r21  r22  tz │ translation│
│  0    0    0   1  │            │
└                                ┘
```

To use it in a game engine:

- **Head bone rotation:** take the upper-left 3×3, convert to your engine's
  rotation type (quaternion / Euler / matrix). The 3 columns are the X / Y / Z
  basis vectors of the face after rotation, so converting to a quaternion is
  one library call (`Quaternion.FromMatrix` in Unity, `Basis()` in Godot,
  `FQuat::MakeFromRotationMatrix` in Unreal).
- **Head bone translation:** `(tx, ty, tz)` is the head position in cm
  relative to the camera (or wherever MediaPipe puts its world origin —
  typically a fixed plane in front of the camera). If you only care about
  rotation, ignore this.

`head_euler` is a precomputed **(pitch, yaw, roll) in degrees**, derived from
the same matrix via the Tait-Bryan ZYX convention. Useful for HUD readouts,
IK constraints, or quick "is the head turned more than 30°" checks.

If you need a quaternion server-side, add this near `matrix_to_euler` in
`detect.py`:

```python
def matrix_to_quat(m):
    r = m[:3, :3]
    t = r.trace()
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (r[2,1] - r[1,2]) * s
        y = (r[0,2] - r[2,0]) * s
        z = (r[1,0] - r[0,1]) * s
    else:
        i = np.argmax(np.diag(r))
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i,i] - r[j,j] - r[k,k])
        q = [0.0]*4
        q[0] = (r[k,j] - r[j,k]) / s        # w
        q[i+1] = 0.25 * s
        q[j+1] = (r[j,i] + r[i,j]) / s
        q[k+1] = (r[k,i] + r[i,k]) / s
        w, x, y, z = q
    return [w, x, y, z]
```

---

## Blendshapes (52 ARKit-style)

These are the same keys Apple's ARKit publishes, so any character rig
prepared for an iPhone face capture will work here unchanged. All values are
clamped to `[0, 1]`, where `0` = neutral and `1` = full expression.

```
_neutral
browDownLeft           browDownRight
browInnerUp
browOuterUpLeft        browOuterUpRight
cheekPuff
cheekSquintLeft        cheekSquintRight
eyeBlinkLeft           eyeBlinkRight
eyeLookDownLeft        eyeLookDownRight
eyeLookInLeft          eyeLookInRight
eyeLookOutLeft         eyeLookOutRight
eyeLookUpLeft          eyeLookUpRight
eyeSquintLeft          eyeSquintRight
eyeWideLeft            eyeWideRight
jawForward             jawLeft                jawOpen                jawRight
mouthClose
mouthDimpleLeft        mouthDimpleRight
mouthFrownLeft         mouthFrownRight
mouthFunnel
mouthLeft              mouthRight
mouthLowerDownLeft     mouthLowerDownRight
mouthPressLeft         mouthPressRight
mouthPucker
mouthRollLower         mouthRollUpper
mouthShrugLower        mouthShrugUpper
mouthSmileLeft         mouthSmileRight
mouthStretchLeft       mouthStretchRight
mouthUpperUpLeft       mouthUpperUpRight
noseSneerLeft          noseSneerRight
```

Quick reference for common driving:

| effect to drive  | use blendshape(s)                                    |
|------------------|------------------------------------------------------|
| open mouth       | `jawOpen`                                            |
| smile            | `mouthSmileLeft + mouthSmileRight` (sum, clamp to 1) |
| frown            | `mouthFrownLeft + mouthFrownRight`                   |
| blink            | `eyeBlinkLeft`, `eyeBlinkRight` (per eye)            |
| gaze direction   | `eyeLook{In,Out,Up,Down}{Left,Right}`                |
| raised eyebrows  | `browInnerUp` + `browOuterUp{Left,Right}`            |
| surprise         | `eyeWide*` + `jawOpen` + `browInnerUp`               |
| pucker (kiss)    | `mouthPucker`                                        |

Tip: keys with `_neutral` are mostly a stability hint — when `_neutral`
approaches 1 the model is telling you the face is relaxed; don't drive any
morphs in that case.

---

## Face oval, contours, tesselation, irises

### Face oval (silhouette)

36 vertices forming a closed loop around the outer face boundary, in this
walk order (extracted at startup by `_face_oval_ordered()`):

```
10 → 338 → 297 → 332 → 284 → 251 → 389 → 356 → 454 → 323 →
361 → 288 → 397 → 365 → 379 → 378 → 400 → 377 → 152 → 148 →
176 → 149 → 150 → 136 → 172 →  58 → 132 →  93 → 234 → 127 →
162 →  21 →  54 → 103 →  67 → 109 → (back to 10)
```

This is what we use as the polygon for the binary face mask
(`_face_mask` in `detect.py`).

### Tesselation triangles

`FACEMESH_TESSELATION` is a Delaunay-style triangulation of all 468 face
landmarks as **2 556 undirected edges**. The script reconstructs the unique
**854 triangles** from those edges once at startup (`_build_triangles()`)
and reuses the list for OBJ export and the UV-atlas warp.

Two facts you can rely on:

1. The triangle list is **topology-invariant**: index `i` of the triangle
   array always refers to the same three landmark indices, no matter who is
   in front of the camera.
2. The triangulation is **a watertight mesh** (no holes, no overlaps) over
   the 468 face landmarks — but it does **not** include the iris ring (468–477).
   Drop iris vertices if you want a closed face surface; keep them if you
   want eye-region detail and accept the small T-junctions there.

### Iris connections

Two 4-edge closed loops around iris centers 468 / 473:

```
LEFT_IRIS  = [(474,475),(475,476),(476,477),(477,474)]
RIGHT_IRIS = [(469,470),(470,471),(471,472),(472,469)]
```

Useful for drawing the iris ring or measuring pupil position relative to the
eye corners.

---

## UV unwrap & static asset export

Pressing `s` writes a self-contained 3D asset bundle to `./exports/`:

| file                   | content                                                              |
|------------------------|----------------------------------------------------------------------|
| `face_TS.obj`          | 468 vertices, 854 triangles, with per-vertex UVs                     |
| `face_TS.mtl`          | material referencing the UV texture                                  |
| `face_TS_uv.png`       | 1024×1024 BGRA UV atlas — the live frame warped into texture space   |
| `face_TS_cutout.png`   | the face crop with a feathered alpha mask (decal use)                |

**How the UV atlas is built** (`_unwrap_atlas` in `detect.py`):

1. Each of the 854 triangles is processed once.
2. For each triangle, the 3 source pixel coordinates (in the live frame) are
   mapped to 3 destination pixel coordinates (in the 1024×1024 atlas, using
   bbox-normalized image coords as the UV layout).
3. `cv2.getAffineTransform` + `cv2.warpAffine` warp the triangle's pixels
   from camera space into atlas space; a per-triangle binary mask prevents
   bleed into neighboring triangles.
4. The OBJ exports `vt` coordinates in the same bbox-normalized layout (with
   the V axis flipped to match OBJ texture-space convention), so the texture
   maps onto the model with zero manual UV work.

> **Caveat:** the UV layout is **pose-dependent**. If the head is tilted when
> you press `s`, the atlas captures that tilt. For consistent textures
> across captures, hold a frontal pose. If you need pose-invariant atlases,
> apply `inverse(head_matrix)` to landmarks before normalizing — left as an
> exercise (or open an issue).

---

## Game engine integration

The shape of the integration is the same regardless of engine: open a TCP
socket, read lines, JSON-parse, apply.

**Unity (C#)** — use `System.Net.Sockets.TcpClient`, read lines via
`StreamReader.ReadLine()`, parse with `JsonUtility` (or Newtonsoft for
nested dicts), then per `FixedUpdate`:

```csharp
headBone.localRotation = QuaternionFromMatrix(packet.head_matrix);
foreach (var kv in packet.blendshapes) {
    int idx = skinnedMesh.sharedMesh.GetBlendShapeIndex(kv.Key);
    if (idx >= 0) skinnedMesh.SetBlendShapeWeight(idx, kv.Value * 100f);
}
```

**Unreal (C++ / Blueprint)** — `FSocket`, `Recv()`, line-buffer, parse
with `FJsonObjectConverter`, drive `UPoseableMeshComponent` rotation and
morph target weights via `SetMorphTarget(FName, float)`.

**Godot (GDScript)** — `StreamPeerTCP`, `get_string(get_available_bytes())`,
buffer-and-split by newline, parse with `JSON.parse_string`, drive a
`Skeleton3D` head bone via `set_bone_pose_rotation`.

The 52 blendshape names are the ARKit ones, so a VRM / MetaHuman / Ready
Player Me character with ARKit morphs works **with zero rigging changes** —
the names align by convention.

---

## Building the .exe

```bat
build.bat
```

This invokes PyInstaller with:

```
--onefile --console --noconfirm
--name face-mesh-stream
--collect-all mediapipe
--add-data "face_landmarker.task;."
--hidden-import mediapipe.framework.formats.landmark_pb2
```

Output: `dist\face-mesh-stream.exe` (~234 MB — Python + mediapipe +
OpenCV + numpy + the model are all bundled).

When run, the `.exe` writes its outputs (`exports/`, `recordings/`,
`videos/`) **next to the .exe itself**, not into the temporary PyInstaller
extraction dir — so you can drop the `.exe` into any folder, double-click,
and find your captures right there.

Prerequisite: run `python detect.py` once first so `face_landmarker.task`
exists in the project folder for PyInstaller to pick up. Or
`pip install pyinstaller` and let the script's autodownload run inside the
build (slower, but works).

---

## What to rely on, what NOT to rely on

✅ **Rely on:**

- **Landmark indices.** They are stable across versions, people, and frames
  — if landmark 1 is the nose tip today, it will be the nose tip tomorrow.
- **Blendshape names.** ARKit's vocabulary; well-defined and supported by
  every major character pipeline.
- **`head_matrix` for rotation.** The orthonormal 3×3 block is reliable;
  decompose to your engine's rotation primitive of choice.
- **The face oval for silhouette/mask work.** Stable, closed, well-defined.

⚠️ **Be careful with:**

- **Raw `z` depth.** It's metric-*ish*, not calibrated. Good for relative
  comparisons within a face, mediocre for absolute distances. If you need
  metric scale, solve PnP yourself with known camera intrinsics.
- **`head_matrix` translation.** Only meaningful relative to MediaPipe's
  internal camera-plane origin; doesn't tell you "head is 50 cm from
  camera" without intrinsics.
- **Tracking near profile views** (yaw > ~70°). Landmarks behind the head
  are extrapolated and can drift visibly. The blendshape head also degrades
  there.
- **Iris position under heavy makeup or contact-lens patterns.** Eye color
  and lid creases can confuse the iris head; the rest of the mesh is fine.

❌ **Don't rely on:**

- **Frame-to-frame stability of every individual landmark.** Small jitter
  (~1–2 px at 720p) is normal. Smooth in your code if you care
  (one-euro filter is the standard answer).
- **Sub-pixel accuracy on closed eyes.** When the eye is closed, the top and
  bottom lid landmarks collapse onto each other and EAR-style ratios drop
  to noise.

---

## Performance notes

- ~15–25 ms / frame on a modern desktop CPU (one inference call, XNNPACK).
- ~30–60 ms / frame on integrated laptop GPUs without a delegate; switching
  to `delegate=Delegate.GPU` in `BaseOptions` halves that on supported
  systems.
- TCP broadcast itself is ~50 µs / client on localhost — negligible.
- The dense (`mode == 3`) wireframe renderer draws ~2 500 lines per frame
  via `mp_drawing.draw_landmarks`; if you ever feel the redraw, that's the
  spot.
- Profile with `python -m cProfile -o p.prof detect.py` and inspect with
  [snakeviz](https://jiffyclub.github.io/snakeviz/).

"""
Realtime face mesh + head pose + blendshape streamer.

Drives a game character with the player's face (FaceID-style developer tool).
On every frame the script emits ONE JSON object over TCP 127.0.0.1:8765:

    {
      "ts":          float,         # unix time
      "ok":          bool,
      "landmarks":   [[x,y,z],...], # 478 normalized points (0..1, z roughly metric)
      "head_matrix": [[...],...,    # 4x4 row-major, face-canonical -> world
                      [...]],
      "head_euler":  [pitch, yaw, roll],  # degrees
      "blendshapes": { "browDownLeft": 0.12, ... }  # 52 ARKit-style keys
    }

Game engine connects, reads lines, applies matrix to head bone and blendshapes
to facial morph targets. Texture is applied by the game on its own model.

Keys:
    1/2/3/4 — render mode (wire / textured / textured+wire / dense)
    i / c   — toggle irises / contours
    p       — toggle pose axes
    r       — toggle JSONL recording -> ./recordings
    v       — toggle MP4 video of the canvas -> ./videos
    s       — export static asset bundle (.obj + .mtl + UV atlas + cutout)
    Esc/q   — quit
"""
import sys
import time
import json
import math
import socket
import threading
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


CAP_W = 1280
CAP_H = 720
TEX_SIZE = 1024

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 8765

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_NAME = "face_landmarker.task"


def app_dir() -> Path:
    """Writable persistent dir — next to .exe when frozen, next to script otherwise."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    """Read-only bundled data dir (PyInstaller _MEIPASS) or app_dir() in dev."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else app_dir()


MODEL_PATH = app_dir() / MODEL_NAME

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

SPEC_TESS = mp_draw.DrawingSpec(color=(80,  80,  80),  thickness=1, circle_radius=0)
SPEC_CONT = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0)
SPEC_IRIS = mp_draw.DrawingSpec(color=(0,   200, 255), thickness=1, circle_radius=1)


# ---- topology ---------------------------------------------------------------

def _face_oval_ordered():
    edges = list(mp_face.FACEMESH_FACE_OVAL)
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    start = edges[0][0]
    seq = [start]; prev = None; cur = start
    while True:
        nxt = next((n for n in adj[cur] if n != prev), None)
        if nxt is None or nxt == start:
            break
        seq.append(nxt); prev, cur = cur, nxt
    return seq


def _build_triangles():
    edges = set()
    for a, b in mp_face.FACEMESH_TESSELATION:
        edges.add((min(a, b), max(a, b)))
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    tris = []
    for a, b in edges:
        for c in adj[a] & adj[b]:
            if c > b:
                tris.append((a, b, c))
    return tris


FACE_OVAL = _face_oval_ordered()
TRIANGLES = _build_triangles()


# ---- model bootstrap --------------------------------------------------------

def ensure_model() -> Path:
    """Locate the model file. Order: next-to-exe → bundled → download next-to-exe."""
    local = app_dir() / MODEL_NAME
    if local.exists() and local.stat().st_size > 100_000:
        return local
    bundled = bundle_dir() / MODEL_NAME
    if bundled.exists() and bundled.stat().st_size > 100_000:
        return bundled
    print(f"downloading model -> {local}", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as resp, open(local, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0; mark = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk: break
                f.write(chunk); read += len(chunk)
                if total and read - mark > 512 * 1024:
                    print(f"  {read/1e6:.1f}/{total/1e6:.1f} MB", flush=True); mark = read
    except Exception as e:
        sys.exit(f"model download failed: {e}\n"
                 f"manually grab {MODEL_URL} -> {local}")
    print("model ready", flush=True)
    return local


def create_detector():
    p = ensure_model()
    base = mp_tasks.BaseOptions(model_asset_path=str(p))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# ---- streaming server -------------------------------------------------------

class StreamServer:
    def __init__(self, host, port):
        self.host = host; self.port = port
        self.clients = []
        self.lock = threading.Lock()
        self._sock = None
        self._stop = False

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(16)
        self._sock = s
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.clients.append(conn)
            print(f"client {addr[0]}:{addr[1]} connected", flush=True)

    def broadcast(self, line):
        payload = (line + "\n").encode("utf-8")
        dead = []
        with self.lock:
            for c in self.clients:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)
                try: c.close()
                except OSError: pass

    def count(self):
        with self.lock:
            return len(self.clients)

    def stop(self):
        self._stop = True
        try: self._sock.close()
        except OSError: pass
        with self.lock:
            for c in self.clients:
                try: c.close()
                except OSError: pass
            self.clients.clear()


# ---- math / drawing ---------------------------------------------------------

def to_normalized_proto(lms):
    p = landmark_pb2.NormalizedLandmarkList()
    for lm in lms:
        v = p.landmark.add()
        v.x = lm.x; v.y = lm.y; v.z = lm.z
    return p


def matrix_to_euler(m):
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw   = math.atan2(-r[2, 0], sy)
        roll  = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw   = math.atan2(-r[2, 0], sy)
        roll  = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def draw_pose_axes(canvas, R, origin, length=80):
    axes = np.eye(3, dtype=np.float32) * length
    rotated = R @ axes
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X red, Y green, Z blue
    ox, oy = int(origin[0]), int(origin[1])
    for i in range(3):
        ex = ox + int(rotated[0, i])
        ey = oy - int(rotated[1, i])
        cv2.line(canvas, (ox, oy), (ex, ey), cols[i], 2, cv2.LINE_AA)


def _face_mask(pts, w, h, feather=0):
    poly = np.array([pts[i] for i in FACE_OVAL], dtype=np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly], 255, lineType=cv2.LINE_AA)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


def render_wire(w, h, proto, *, tess=False, cont=True, iris=True):
    c = np.zeros((h, w, 3), dtype=np.uint8)
    if tess:
        mp_draw.draw_landmarks(c, proto, mp_face.FACEMESH_TESSELATION, None, SPEC_TESS)
    if cont:
        mp_draw.draw_landmarks(c, proto, mp_face.FACEMESH_CONTOURS,    None, SPEC_CONT)
    if iris:
        mp_draw.draw_landmarks(c, proto, mp_face.FACEMESH_IRISES,      None, SPEC_IRIS)
    return c


def render_tex(frame, pts):
    h, w = frame.shape[:2]
    m = _face_mask(pts, w, h, feather=0)
    return cv2.bitwise_and(frame, frame, mask=m)


# ---- static export (still useful for offline rig + texture) -----------------

def _extract_cutout(frame, pts, pad=8):
    h, w = frame.shape[:2]
    m = _face_mask(pts, w, h, feather=4)
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = m
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    x0 = max(x0 - pad, 0); y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, w); y1 = min(y1 + pad, h)
    return bgra[y0:y1, x0:x1].copy()


def _uv_norm(pts):
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    rng = np.maximum(mx - mn, 1.0)
    return (pts - mn) / rng


def _unwrap_atlas(frame, pts, size=TEX_SIZE):
    uv = _uv_norm(pts) * (size - 1)
    atlas = np.zeros((size, size, 3), dtype=np.uint8)
    mask  = np.zeros((size, size),    dtype=np.uint8)
    H, W = frame.shape[:2]
    for a, b, c in TRIANGLES:
        src = np.array([pts[a], pts[b], pts[c]], dtype=np.float32)
        dst = np.array([uv[a],  uv[b],  uv[c]],  dtype=np.float32)
        sx, sy, sw, sh = cv2.boundingRect(src)
        dx, dy, dw, dh = cv2.boundingRect(dst)
        if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0: continue
        if sx < 0 or sy < 0 or sx + sw > W or sy + sh > H: continue
        if dx < 0 or dy < 0: continue
        sl = src - np.float32([sx, sy])
        dl = dst - np.float32([dx, dy])
        M = cv2.getAffineTransform(sl, dl)
        warped = cv2.warpAffine(frame[sy:sy+sh, sx:sx+sw], M, (dw, dh),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        tm = np.zeros((dh, dw), dtype=np.uint8)
        cv2.fillConvexPoly(tm, dl.astype(np.int32), 255, lineType=cv2.LINE_AA)
        y1 = min(dy + dh, size); x1 = min(dx + dw, size)
        h2 = y1 - dy; w2 = x1 - dx
        if h2 <= 0 or w2 <= 0: continue
        roi = atlas[dy:y1, dx:x1]; rmask = mask[dy:y1, dx:x1]
        tmc = tm[:h2, :w2]; wpc = warped[:h2, :w2]
        sel = tmc > 0
        roi[sel] = wpc[sel]; rmask[sel] = tmc[sel]
    return atlas, mask


def export_bundle(out_dir, frame, lms, w, h):
    pts   = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
    pts3d = np.array([[p.x, p.y, p.z]    for p in lms], dtype=np.float32)
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = out_dir / f"face_{ts}"

    cutout = _extract_cutout(frame, pts)
    cutout_path = base.with_name(base.name + "_cutout.png")
    cv2.imwrite(str(cutout_path), cutout)

    atlas, amask = _unwrap_atlas(frame, pts)
    atlas_bgra = cv2.cvtColor(atlas, cv2.COLOR_BGR2BGRA)
    atlas_bgra[:, :, 3] = amask
    tex_path = base.with_name(base.name + "_uv.png")
    cv2.imwrite(str(tex_path), atlas_bgra)

    obj_path = base.with_suffix(".obj")
    mtl_path = base.with_suffix(".mtl")
    uv_n = _uv_norm(pts)
    obj_lines = [f"mtllib {mtl_path.name}", f"o {base.name}"]
    for x, y, z in pts3d:
        obj_lines.append(f"v {x:.6f} {-y:.6f} {-z:.6f}")
    for u, v in uv_n:
        obj_lines.append(f"vt {u:.6f} {1.0 - v:.6f}")
    obj_lines.append("usemtl face_mat")
    for a, b, c in TRIANGLES:
        obj_lines.append(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}")
    obj_path.write_text("\n".join(obj_lines), encoding="utf-8")
    mtl_path.write_text(
        "newmtl face_mat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
        f"map_Kd {tex_path.name}\n", encoding="utf-8",
    )
    return ts, [obj_path, mtl_path, tex_path, cutout_path]


# ---- main loop --------------------------------------------------------------

MODE_NAMES = ["wireframe", "textured", "textured+wire", "dense"]


def run(cam_idx=0):
    cam = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cam.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cam.set(cv2.CAP_PROP_FPS,          30)
    if not cam.isOpened():
        sys.exit("камера не открылась")

    detector = create_detector()
    server = StreamServer(STREAM_HOST, STREAM_PORT)
    server.start()
    print(f"streaming on tcp://{STREAM_HOST}:{STREAM_PORT} (newline-delimited JSON)", flush=True)

    vcam = None
    vcam_w = vcam_h = 0
    try:
        import pyvirtualcam  # type: ignore
        vcam = pyvirtualcam.Camera(width=CAP_W, height=CAP_H, fps=30,
                                   fmt=pyvirtualcam.PixelFormat.RGB)
        vcam_w, vcam_h = CAP_W, CAP_H
        print(f"virtual camera: {vcam.device}", flush=True)
    except Exception as e:
        print(f"virtual camera disabled ({e})", flush=True)

    base_dir = app_dir()
    out_dir = base_dir / "exports";    out_dir.mkdir(exist_ok=True)
    rec_dir = base_dir / "recordings"; rec_dir.mkdir(exist_ok=True)
    vid_dir = base_dir / "videos";     vid_dir.mkdir(exist_ok=True)
    print(f"output dirs:\n  {out_dir}\n  {rec_dir}\n  {vid_dir}", flush=True)

    mode = 0
    show_iris = True
    show_cont = True
    show_pose = True

    recording = False
    rec_file = None
    rec_count = 0
    rec_start = 0.0

    video_writer = None
    video_path = None
    video_count = 0
    video_start = 0.0

    t0 = time.time()
    last_ts_ms = -1
    t_mark = time.time(); n_since = 0; fps = 0.0
    notice = ""; notice_until = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            result = detector.detect_for_video(mp_image, ts_ms)

            canvas = np.zeros_like(frame)
            lms = result.face_landmarks[0] if result.face_landmarks else None

            euler = (0.0, 0.0, 0.0)
            xform = None
            if lms is not None:
                pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
                proto = to_normalized_proto(lms)

                if mode == 0:
                    canvas = render_wire(w, h, proto, tess=False, cont=show_cont, iris=show_iris)
                elif mode == 1:
                    canvas = render_tex(frame, pts)
                elif mode == 2:
                    canvas = render_tex(frame, pts)
                    if show_cont:
                        mp_draw.draw_landmarks(canvas, proto, mp_face.FACEMESH_CONTOURS, None, SPEC_CONT)
                    if show_iris:
                        mp_draw.draw_landmarks(canvas, proto, mp_face.FACEMESH_IRISES,   None, SPEC_IRIS)
                elif mode == 3:
                    canvas = render_wire(w, h, proto, tess=True, cont=show_cont, iris=show_iris)

                if result.facial_transformation_matrixes:
                    xform = result.facial_transformation_matrixes[0]
                    euler = matrix_to_euler(xform)
                    if show_pose:
                        draw_pose_axes(canvas, xform[:3, :3], pts[1], length=90)

                bs = {}
                if result.face_blendshapes:
                    bs = {b.category_name: round(float(b.score), 4) for b in result.face_blendshapes[0]}

                packet = {
                    "ts": time.time(),
                    "ok": True,
                    "landmarks": [[p.x, p.y, p.z] for p in lms],
                    "head_euler": [round(e, 3) for e in euler],
                    "head_matrix": xform.tolist() if xform is not None else None,
                    "blendshapes": bs,
                }
                line = json.dumps(packet, separators=(',', ':'))
                server.broadcast(line)
                if recording and rec_file:
                    rec_file.write(line + "\n")
                    rec_count += 1
            else:
                server.broadcast(json.dumps({"ts": time.time(), "ok": False}))

            n_since += 1
            dt = time.time() - t_mark
            if dt >= 0.5:
                fps = n_since / dt
                t_mark = time.time(); n_since = 0

            head_str = (f"head [P{euler[0]:+6.1f} Y{euler[1]:+6.1f} R{euler[2]:+6.1f}]"
                        if lms is not None else "no face")
            rec_str = f"REC {rec_count}f ({time.time()-rec_start:.1f}s)" if recording else ""
            vid_str = f"VID {video_count}f ({time.time()-video_start:.1f}s)" if video_writer is not None else ""
            hint = (f"{fps:.1f}fps  mode={MODE_NAMES[mode]}  clients={server.count()}  {rec_str} {vid_str}  "
                    f"[1-4 mode | i iris | c cont | p pose | r rec | v vid | s save | Esc quit]")

            for txt, y in ((head_str, 30), (hint, h - 8)):
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (4, y - th - 6), (12 + tw, y + 4), (0, 0, 0), -1)
                cv2.putText(canvas, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1, cv2.LINE_AA)

            if time.time() < notice_until:
                (tw, th), _ = cv2.getTextSize(notice, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(canvas, (4, 50), (12 + tw, 58 + th), (0, 100, 0), -1)
                cv2.putText(canvas, notice, (8, 54 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)

            if video_writer is not None:
                video_writer.write(canvas)
                video_count += 1

            cv2.imshow("face mesh stream", canvas)

            if vcam is not None:
                try:
                    out = canvas
                    if w != vcam_w or h != vcam_h:
                        out = cv2.resize(canvas, (vcam_w, vcam_h), interpolation=cv2.INTER_LINEAR)
                    vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    vcam.sleep_until_next_frame()
                except Exception:
                    pass

            k = cv2.waitKey(1) & 0xFFFF
            if k in (27, ord('q'), 0x0439):
                break
            elif k == ord('1'): mode = 0
            elif k == ord('2'): mode = 1
            elif k == ord('3'): mode = 2
            elif k == ord('4'): mode = 3
            elif k in (ord('i'), 0x0448): show_iris = not show_iris
            elif k in (ord('c'), 0x0441): show_cont = not show_cont
            elif k in (ord('p'), 0x0437): show_pose = not show_pose
            elif k in (ord('r'), 0x043A):
                if recording:
                    rec_file.close()
                    notice = f"recording stopped: {rec_count} frames"
                    notice_until = time.time() + 2.0
                    print(f"recording saved: {rec_file.name} ({rec_count} frames)", flush=True)
                    recording = False; rec_file = None
                else:
                    fn = rec_dir / f"rec_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
                    rec_file = open(fn, "w", encoding="utf-8", buffering=1)
                    rec_count = 0; rec_start = time.time(); recording = True
                    notice = f"recording -> {fn.name}"
                    notice_until = time.time() + 2.0
                    print(f"recording to {fn}", flush=True)
            elif k in (ord('s'), 0x044B):
                if lms is not None:
                    ts, files = export_bundle(out_dir, frame, lms, w, h)
                    notice = f"saved face_{ts} bundle"
                    notice_until = time.time() + 2.0
                    print(notice, flush=True)
                    for f in files:
                        print(f"  {f}", flush=True)
                else:
                    notice = "no face detected"
                    notice_until = time.time() + 2.0
            elif k in (ord('v'), 0x043C):
                if video_writer is not None:
                    video_writer.release()
                    notice = f"video stopped: {video_count} frames -> {video_path.name}"
                    notice_until = time.time() + 2.5
                    print(f"video saved: {video_path} ({video_count} frames)", flush=True)
                    video_writer = None; video_path = None
                else:
                    video_path = vid_dir / f"video_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    target_fps = max(fps, 15.0) if fps > 0 else 30.0
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, target_fps, (w, h))
                    if not video_writer.isOpened():
                        video_writer = None
                        notice = "video: codec mp4v unavailable"
                        notice_until = time.time() + 2.5
                    else:
                        video_count = 0; video_start = time.time()
                        notice = f"video -> {video_path.name}"
                        notice_until = time.time() + 2.5
                        print(f"recording video to {video_path}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()
        detector.close()
        server.stop()
        if vcam is not None:
            try: vcam.close()
            except Exception: pass
        if rec_file is not None:
            try: rec_file.close()
            except Exception: pass
        if video_writer is not None:
            try: video_writer.release()
            except Exception: pass


if __name__ == "__main__":
    run()

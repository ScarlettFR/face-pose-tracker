"""
Minimal TCP client for the face-mesh stream.
Connects to detect.py and prints one summary line per received frame.

Run detect.py in one terminal, then:
    python client_example.py

In Unity / Godot / Unreal you'd do the equivalent: open TCP socket to
127.0.0.1:8765, read lines, json-parse each, and apply:
  - head_matrix  -> head bone transform
  - blendshapes  -> facial morph target weights
  - landmarks    -> optional per-vertex deformation
"""
import socket
import json

HOST, PORT = "127.0.0.1", 8765


def main():
    with socket.create_connection((HOST, PORT)) as s:
        f = s.makefile("r", encoding="utf-8")
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pkt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not pkt.get("ok"):
                print("no face", flush=True)
                continue
            p, y, r = pkt["head_euler"]
            bs = pkt["blendshapes"]
            jaw   = bs.get("jawOpen",     0.0)
            smile = bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)
            blink = bs.get("eyeBlinkLeft",   0.0) + bs.get("eyeBlinkRight",   0.0)
            print(f"P{p:+6.1f} Y{y:+6.1f} R{r:+6.1f}  "
                  f"jaw={jaw:.2f} smile={smile:.2f} blink={blink:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()

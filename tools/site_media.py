"""Camera frames -> docs/assets/media/* for the project site. Training-time tool (needs ffmpeg and ImageMagick).

  python tools/site_media.py hero RAW.bgra [--fps 20] [--size 1920x1080] [--crf 29]
  python tools/site_media.py stills CAPTURE_DIR

hero: a raw BGRA file written by `tools/capture.py --raw --window T0:T1 --size 1920x1080 --every 0.05` (real time at 20
  frames/s) -> hero.mp4 (H.264, no audio) and hero-poster.jpg. The loop is seamless: the clip's last second is crossfaded
  into its first, so the video can simply `loop`. One file only: VP9 came out larger than H.264 at the same quality.
stills: the frames tools/pick_stills.py picks for each named corner (second lap) -> still-<corner>.jpg.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "media")
sys.path.insert(0, os.path.join(ROOT, "tools"))


def hero(raw, fps, size, crf):
    n = os.path.getsize(raw) // (int(size.split("x")[0]) * int(size.split("x")[1]) * 4)
    secs = n / fps
    src = ["-f", "rawvideo", "-pix_fmt", "bgra", "-video_size", size, "-framerate", str(fps), "-i", raw]
    fc = ("[0:v]format=yuv420p,split=3[a][b][c];"
          "[a]trim=start=1:end=%g,setpts=PTS-STARTPTS[m];"
          "[b]trim=start=%g:end=%g,setpts=PTS-STARTPTS[t];"
          "[c]trim=start=0:end=1,setpts=PTS-STARTPTS[h];"
          "[t][h]blend=all_expr='A*(1-T)+B*T'[x];[m][x]concat=n=2:v=1:a=0[v]") % (secs - 1, secs - 1, secs)
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + src + ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-preset",
                    "slow", "-crf", str(crf), "-movflags", "+faststart", "-an", os.path.join(OUT, "hero.mp4")], check=True)
    # the poster is the clip's first frame (the frame at 1 s of the capture)
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + src + ["-ss", "1", "-frames:v", "1", "-vf", "scale=1600:-1", "-q:v", "4",
                    os.path.join(OUT, "hero-poster.jpg")], check=True)
    for f in ("hero.mp4", "hero-poster.jpg"):
        print("%s %d bytes" % (f, os.path.getsize(os.path.join(OUT, f))))


def stills(capdir):
    from pick_stills import pick
    for name, (frame, t, v, ds, n) in pick(capdir, "chase", 2).items():
        out = os.path.join(OUT, "still-%s.jpg" % name.replace("_", "-"))
        subprocess.run(["convert", os.path.join(capdir, frame), "-strip", "-quality", "80", out], check=True)
        print("%s <- %s (t=%.1f s, %.1f m/s)" % (os.path.basename(out), frame, t, v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["hero", "stills"])
    ap.add_argument("src")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--crf", type=int, default=29)
    a = ap.parse_args()
    hero(a.src, a.fps, a.size, a.crf) if a.what == "hero" else stills(a.src)

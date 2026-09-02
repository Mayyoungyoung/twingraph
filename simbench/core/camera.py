"""Upright camera capture, mp4 recording and a live viewer for simbench.

The old robosuite path read pixels through ``mjr_readPixels`` without the
vertical flip mujoco >= 2.0 needs, so every recorded frame was upside
down.  Here offscreen capture goes through :class:`mujoco.Renderer`,
which returns upright images directly; the live viewer is
``mujoco.viewer.launch_passive``.

Usage:
    cam = Camera(ctx, "agentview", 640, 480)
    img = cam.capture()                      # upright HxWx3 uint8

    rec = VideoRecorder(ctx, "out.mp4", camera="agentview", every=5)
    ctx.on_control_step = rec                # sample frames while running
    ... run skills ...
    rec.close()

    viewer = LiveViewer(ctx, realtime=True)  # needs a DISPLAY
    ctx.on_control_step = viewer

VideoRecorder notes:
  * frames are pushed into a queue and encoded by a DAEMON thread --
    the capture hook only renders and enqueues, so encoding never
    stalls the sim loop (the old synchronous writer stretched the
    video wall time and played jittery);
  * call ``set_stage(label)`` before each skill and
    ``add_result(label, ok, detail)`` after it to annotate the
    sub-task banner (top) and the per-stage pass/fail ticker (bottom).
"""
import os
import queue
import threading
import time

import numpy as np


class Camera:
    """Offscreen capture from a scene camera (upright)."""

    def __init__(self, ctx, camera_name="agentview", width=640, height=480):
        import mujoco
        self._mujoco = mujoco
        self.ctx = ctx
        self.name = camera_name
        self.width, self.height = width, height
        self._renderer = None      # created lazily (EGL context)

    def _ensure(self):
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.ctx.model, height=self.height, width=self.width)
        return self._renderer

    def capture(self):
        r = self._ensure()
        r.update_scene(self.ctx.data, camera=self.name)
        return r.render()          # upright HxWx3 uint8


class VideoRecorder:
    """Sample one annotated frame every ``every`` control steps and
    encode asynchronously (no frame drops / no wall-time stretch).

    Attach as ``ctx.on_control_step``; skills call ``set_stage`` /
    ``add_result`` for the overlay text.  ``close`` drains the encode
    queue (waits up to 30s).
    """

    def __init__(self, ctx, path, camera="agentview", every=5, fps=30,
                 width=640, height=480, title="", repeat=1):
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        self.ctx = ctx
        self.cam = Camera(ctx, camera, width, height)
        self.every = max(1, every)
        self.repeat = max(1, repeat)   # slow motion: write each
                                       # sampled frame this many times
                                       # (stretches playback, keeps fps)
        self.title = title
        self.stage = "init"
        self.history = []          # (label, ok, detail)
        self.n = 0
        self.t0 = time.time()
        self.frames = 0
        # async encode: render+enqueue only, a daemon thread writes the
        # mp4 -- the synchronous writer stretched wall time with the
        # encode cost (jittery playback at mismatched speed).
        self._q = queue.Queue(maxsize=600)
        self._writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                          quality=8)
        self._enc = threading.Thread(target=self._encode_loop, daemon=True)
        self._enc.start()
        try:
            from PIL import ImageFont
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            self.font_s = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            from PIL import ImageFont
            self.font = self.font_s = ImageFont.load_default()

    # ----------------------------------------------------------- overlay
    def set_stage(self, label):
        self.stage = label

    def add_result(self, label, ok, detail=""):
        self.history.append((label, bool(ok), detail))

    # ------------------------------------------------------------- hooks
    def __call__(self, ctx):
        self.n += 1
        if self.n % self.every:
            return
        img = self._annotate(self.cam.capture())
        for _ in range(self.repeat):
            try:
                self._q.put_nowait(img)
            except queue.Full:
                pass               # encoder falling behind: drop oldest
        self.frames += 1

    def _encode_loop(self):
        while True:
            img = self._q.get()
            if img is None:
                break
            self._writer.append_data(img)

    def _annotate(self, img):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        w, h = im.size
        # top banner: task + current sub-task
        hdr = f"{self.title}   [{self.stage}]"
        d.rectangle([0, 0, w, 30], fill=(10, 10, 10))
        d.text((8, 4), hdr, fill=(255, 255, 0), font=self.font)
        # bottom: ONLY the stage verdicts (S1..ST) -- the raw skill
        # pass/fail ticker is filtered out (the user wants the video to
        # show just the current task's success/failure)
        stages = [(lb, ok) for lb, ok, _ in self.history
                  if lb.startswith("S")]
        hist = "  ".join(f"{lb}:{'ok' if ok else 'FAIL'}"
                         for lb, ok in stages)
        d.rectangle([0, h - 26, w, h], fill=(10, 10, 10))
        d.text((8, h - 24), hist, fill=(0, 255, 120), font=self.font_s)
        # elapsed sim time (top-right)
        el = f"t={time.time() - self.t0:5.1f}s  n={self.n}"
        d.text((w - 210, 4), el, fill=(200, 200, 200), font=self.font_s)
        return np.array(im)

    def close(self):
        self._q.put(None)          # sentinel: stop the encoder
        self._enc.join(timeout=30.0)
        self._writer.close()


class LiveViewer:
    """Live passive viewer synced on control steps.

    mujoco >= 2.3.3 uses ``viewer.launch_passive`` (with .sync()).  mujoco
    2.3.2 (this repo's env) lacks it, so the Simulate GUI runs in a daemon
    thread with ``run_physics_thread=False``: the GUI only renders the
    shared MjData while our control loop is the sole stepper.

    realtime=True throttles the sim to wall-clock speed; otherwise the sim
    runs as fast as possible and the viewer samples the motion.  Falls back
    to a no-op with a warning when no DISPLAY is available.
    """

    def __init__(self, ctx, realtime=False, sync_every=1):
        self.ctx = ctx
        self.realtime = realtime
        self.sync_every = max(1, sync_every)
        self.n = 0
        self._last = time.time()
        self._viewer = None
        self._thread = None
        try:
            import mujoco.viewer
            if hasattr(mujoco.viewer, "launch_passive"):
                self._viewer = mujoco.viewer.launch_passive(
                    ctx.model, ctx.data)
            else:
                # mujoco 2.3.2 path
                self._thread = threading.Thread(
                    target=mujoco.viewer.launch,
                    args=(ctx.model, ctx.data),
                    kwargs=dict(run_physics_thread=False),
                    daemon=True)
                self._thread.start()
                time.sleep(1.0)     # let the GUI load the model first
        except Exception as e:                    # no DISPLAY / headless
            print(f"[LiveViewer] unavailable ({e}); continuing headless. "
                  f"Use --record for videos.")

    def __call__(self, ctx):
        self.n += 1
        if self.realtime:
            now = time.time()
            budget = self.ctx.control_dt
            elapsed = now - self._last
            if elapsed < budget:
                time.sleep(budget - elapsed)
            self._last = time.time()
        if self._viewer is not None and self.n % self.sync_every == 0:
            self._viewer.sync()

    @property
    def hard_exit(self):
        """True when a 2.3.2 daemon-thread GUI is still open: interpreter
        shutdown would abort (atexit glfw.terminate racing the render
        loop), so the runner should ``os._exit`` instead."""
        return self._thread is not None and self._thread.is_alive()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        # 2.3.2 daemon-thread GUI: closed by the user's window or at exit

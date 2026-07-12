"""Interactively select fixed-size metasurface ROIs."""

from __future__ import annotations

import argparse
import base64
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser

import numpy as np

from ..config.roi import MetasurfaceRoi, fixed_roi_from_center, save_rois_json, validate_rois


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("ROI selection requires opencv-python") from exc
    return cv2


def _load_dataset_defaults(dataset_dir: Path, frame_index: int) -> tuple[Path, MetasurfaceRoi]:
    report_path = dataset_dir / "calibration_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frames = report.get("cube_frames", [])
    if not frames:
        raise ValueError(f"{report_path} has no cube_frames")
    frame_index = max(0, min(int(frame_index), len(frames) - 1))
    image_path = Path(frames[frame_index]["image"])
    roi = report["roi"]
    base_roi = MetasurfaceRoi("roi0", int(roi["x0"]), int(roi["y0"]), int(roi["size"]), int(roi["size"]))
    return image_path, base_roi


def _draw_rois(image, rois, active_message: str):
    cv2 = _require_cv2()
    canvas = image.copy()
    for index, roi in enumerate(rois):
        color = (0, 0, 255) if index == 0 else (0, 255, 255)
        cv2.rectangle(canvas, (roi.x0, roi.y0), (roi.x1 - 1, roi.y1 - 1), color, 3)
        cv2.drawMarker(
            canvas,
            (roi.x0 + roi.width // 2, roi.y0 + roi.height // 2),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )
        cv2.putText(
            canvas,
            roi.name,
            (roi.x0, max(20, roi.y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(canvas, active_message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _write_selection(image, image_path: Path, rois, output_path: Path, overlay_path: Path | None = None):
    cv2 = _require_cv2()
    save_rois_json(output_path, rois, image_path=str(image_path))
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(overlay_path), _draw_rois(image, rois, "saved metasurface ROIs"))
        if not ok:
            raise ValueError(f"failed to write overlay: {overlay_path}")
    print(f"saved {output_path}")


def _select_rois_cv2(image, image_path: Path, base_roi: MetasurfaceRoi, output_path: Path, overlay_path: Path | None = None):
    cv2 = _require_cv2()
    image_height, image_width = image.shape[:2]
    rois = [base_roi]
    validate_rois(rois, image_width=image_width, image_height=image_height)
    window = "select 3 extra 160x160 metasurface ROIs"

    def on_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN or len(rois) >= 4:
            return
        rois.append(
            fixed_roi_from_center(
                f"roi{len(rois)}",
                x,
                y,
                image_width=image_width,
                image_height=image_height,
            )
        )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1400, image_width), min(900, image_height))
    cv2.setMouseCallback(window, on_mouse)
    while True:
        message = "click centers for 3 extra ROIs | u undo | s save | q quit"
        canvas = _draw_rois(image, rois, message)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("u") and len(rois) > 1:
            rois.pop()
        elif key == ord("s"):
            if len(rois) != 4:
                print(f"need exactly 4 ROIs including roi0, currently {len(rois)}")
                continue
            break
        elif key == ord("q") or key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("ROI selection cancelled")
    cv2.destroyWindow(window)

    _write_selection(image, image_path, rois, output_path, overlay_path)
    return rois


def _image_data_url(image) -> str:
    cv2 = _require_cv2()
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("failed to encode image for browser selector")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _browser_html(image, rois, output_path: Path) -> bytes:
    image_height, image_width = image.shape[:2]
    initial_rois = [roi.to_dict() for roi in rois]
    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Metasurface ROI selector</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #202124; color: #f1f3f4; }}
.bar {{ position: sticky; top: 0; z-index: 2; padding: 10px 14px; background: #111; }}
button {{ margin-right: 8px; padding: 7px 12px; }}
#wrap {{ padding: 12px; }}
canvas {{ max-width: calc(100vw - 24px); height: auto; background: #000; cursor: crosshair; }}
code {{ color: #9ccaff; }}
</style>
</head>
<body>
<div class="bar">
  Click centers for 3 extra 160x160 ROIs. Red is roi0. Yellow are new ROIs.
  <button onclick="undoRoi()">Undo</button>
  <button onclick="saveRois()">Save</button>
  <span id="status"></span><br>
  Output: <code>{html.escape(str(output_path))}</code>
</div>
<div id="wrap"><canvas id="canvas"></canvas></div>
<script>
const imageWidth = {int(image_width)};
const imageHeight = {int(image_height)};
const roiSize = 160;
const imageData = {json.dumps(_image_data_url(image))};
let rois = {json.dumps(initial_rois)};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const img = new Image();

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function fixedRoiFromCenter(name, cx, cy) {{
  const x0 = clamp(Math.round(cx - roiSize / 2), 0, imageWidth - roiSize);
  const y0 = clamp(Math.round(cy - roiSize / 2), 0, imageHeight - roiSize);
  return {{name, x0, y0, width: roiSize, height: roiSize}};
}}
function draw() {{
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = 8;
  ctx.font = '36px Arial';
  for (let i = 0; i < rois.length; i++) {{
    const r = rois[i];
    ctx.strokeStyle = i === 0 ? '#ff3030' : '#ffe94a';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.strokeRect(r.x0, r.y0, r.width, r.height);
    ctx.fillText(r.name, r.x0, Math.max(42, r.y0 - 12));
    ctx.strokeStyle = '#38a8ff';
    const cx = r.x0 + r.width / 2;
    const cy = r.y0 + r.height / 2;
    ctx.beginPath();
    ctx.moveTo(cx - 18, cy); ctx.lineTo(cx + 18, cy);
    ctx.moveTo(cx, cy - 18); ctx.lineTo(cx, cy + 18);
    ctx.stroke();
  }}
  statusEl.textContent = ` selected ${{rois.length}}/4`;
}}
function canvasPoint(evt) {{
  const rect = canvas.getBoundingClientRect();
  return {{
    x: (evt.clientX - rect.left) * canvas.width / rect.width,
    y: (evt.clientY - rect.top) * canvas.height / rect.height
  }};
}}
canvas.addEventListener('click', (evt) => {{
  if (rois.length >= 4) return;
  const p = canvasPoint(evt);
  rois.push(fixedRoiFromCenter(`roi${{rois.length}}`, p.x, p.y));
  draw();
}});
function undoRoi() {{
  if (rois.length > 1) {{
    rois.pop();
    draw();
  }}
}}
async function saveRois() {{
  if (rois.length !== 4) {{
    alert(`Need exactly 4 ROIs including roi0; currently ${{rois.length}}.`);
    return;
  }}
  const response = await fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{rois}})
  }});
  const text = await response.text();
  if (!response.ok) {{
    alert(text);
    return;
  }}
  statusEl.textContent = ' saved. You can close this tab.';
}}
document.addEventListener('keydown', (evt) => {{
  if (evt.key === 'u') undoRoi();
  if (evt.key === 's') saveRois();
}});
img.onload = () => {{
  canvas.width = imageWidth;
  canvas.height = imageHeight;
  draw();
}};
img.src = imageData;
</script>
</body>
</html>
"""
    return body.encode("utf-8")


def _select_rois_browser(
    image,
    image_path: Path,
    base_roi: MetasurfaceRoi,
    output_path: Path,
    overlay_path: Path | None = None,
    port: int = 0,
    open_browser: bool = True,
):
    image_height, image_width = image.shape[:2]
    rois = [base_roi]
    validate_rois(rois, image_width=image_width, image_height=image_height)
    saved = {"rois": None, "error": None}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            payload = _browser_html(image, rois, output_path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                selected = [MetasurfaceRoi.from_dict(item) for item in data["rois"]]
                selected = validate_rois(selected, image_width=image_width, image_height=image_height)
                if len(selected) != 4:
                    raise ValueError(f"need exactly 4 ROIs, got {len(selected)}")
                _write_selection(image, image_path, selected, output_path, overlay_path)
                saved["rois"] = selected
                payload = b"saved"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:  # report to browser and caller
                saved["error"] = str(exc)
                payload = str(exc).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            finally:
                done.set()

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Browser ROI selector: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        done.wait()
    finally:
        server.shutdown()
        server.server_close()
    if saved["error"] is not None:
        raise RuntimeError(saved["error"])
    return saved["rois"]


def select_rois(
    image_path: Path,
    base_roi: MetasurfaceRoi,
    output_path: Path,
    overlay_path: Path | None = None,
    mode: str = "auto",
    port: int = 0,
    open_browser: bool = True,
):
    cv2 = _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    if mode not in ("auto", "gui", "browser"):
        raise ValueError("mode must be auto, gui, or browser")
    if mode in ("auto", "gui"):
        try:
            return _select_rois_cv2(image, image_path, base_roi, output_path, overlay_path)
        except cv2.error as exc:
            if mode == "gui":
                raise
            print(f"OpenCV GUI is unavailable; falling back to browser selector. Root error: {exc}")
    return _select_rois_browser(
        image,
        image_path,
        base_roi,
        output_path,
        overlay_path,
        port=port,
        open_browser=open_browser,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, help="Existing real dataset dir with calibration_report.json")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--image", type=Path, help="Representative image path; overrides --dataset-dir image")
    parser.add_argument("--base-roi", nargs=2, type=int, metavar=("X0", "Y0"), help="Base ROI top-left if --dataset-dir is not used")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--mode", choices=("auto", "gui", "browser"), default="auto")
    parser.add_argument("--port", type=int, default=0, help="Browser selector port; 0 chooses a free port")
    parser.add_argument("--no-open-browser", action="store_true", help="Print the local URL without opening a browser")
    return parser


def main():
    args = build_parser().parse_args()
    if args.dataset_dir is not None:
        image_path, base_roi = _load_dataset_defaults(args.dataset_dir, args.frame_index)
    else:
        if args.image is None or args.base_roi is None:
            raise ValueError("without --dataset-dir, pass --image and --base-roi X0 Y0")
        image_path = args.image
        base_roi = MetasurfaceRoi("roi0", int(args.base_roi[0]), int(args.base_roi[1]))
    if args.image is not None:
        image_path = args.image
    select_rois(
        image_path,
        base_roi,
        args.output,
        args.overlay,
        mode=args.mode,
        port=args.port,
        open_browser=not args.no_open_browser,
    )


if __name__ == "__main__":
    main()

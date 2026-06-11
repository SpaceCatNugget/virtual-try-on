import os
import json
import time
import uuid
import shutil
import subprocess
import threading
import requests
from io import BytesIO
from pathlib import Path

# Load .env for local development (ignored if not installed / not present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Compatibility shim — newer huggingface_hub removed HfFolder which old gradio needs
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return _hfhub.get_token()
        @staticmethod
        def save_token(token): pass
    _hfhub.HfFolder = _HfFolder

import spaces
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------------------------
# Paths — use /data (persistent storage) if available, else /home/user
# ---------------------------------------------------------------------------
BASE_DIR = "/data" if os.path.exists("/data") else "/home/user"
COMFYUI_DIR = f"{BASE_DIR}/ComfyUI"
COMFYUI_INPUT = f"{COMFYUI_DIR}/input"
COMFYUI_OUTPUT = f"{COMFYUI_DIR}/output"
COMFYUI_MODELS = f"{COMFYUI_DIR}/models"
COMFYUI_CUSTOM_NODES = f"{COMFYUI_DIR}/custom_nodes"
COMFYUI_URL = "http://127.0.0.1:8188"

# ---------------------------------------------------------------------------
# Custom nodes  
# ---------------------------------------------------------------------------
CUSTOM_NODES = {
    "ComfyUI-GGUF":                 "https://github.com/city96/ComfyUI-GGUF",
    "masquerade-nodes-comfyui":     "https://github.com/BadCafeCode/masquerade-nodes-comfyui",
    "ComfyUI-KJNodes":              "https://github.com/kijai/ComfyUI-KJNodes",
    "ComfyUI_LayerStyle_Advance":   "https://github.com/chflame163/ComfyUI_LayerStyle_Advance",
    "comfyui-sam3":                 "https://github.com/PozzettiAndrea/ComfyUI-SAM3",
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = [
    {
        "repo_id":  "unsloth/FLUX.2-klein-4B-GGUF",
        "filename": "flux-2-klein-4b-Q8_0.gguf",
        "dest":     f"{COMFYUI_MODELS}/unet/flux-2-klein-4b-Q8_0.gguf",
    },
    {
        "repo_id":  "Comfy-Org/z_image_turbo",
        "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
        "dest":     f"{COMFYUI_MODELS}/text_encoders/qwen_3_4b.safetensors",
    },
    {
        "repo_id":  "Comfy-Org/flux2-dev",
        "filename": "split_files/vae/flux2-vae.safetensors",
        "dest":     f"{COMFYUI_MODELS}/vae/flux2-vae.safetensors",
    },
    {
        "repo_id":  "p1atdev/auraflow-v0.3-pvc-style-lora",
        "filename": "aura-pvc-2-_00010e_074520s.safetensors",
        "revision": "cafeee8ab8681ab679944b4e75ab0bdc4bdec6f7",
        "dest":     f"{COMFYUI_MODELS}/loras/aura-pvc-2-_00010e_074520s.safetensors",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: str, **kwargs):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def download_model(model: dict):
    dest = Path(model["dest"])
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...")
    kwargs = dict(repo_id=model["repo_id"], filename=model["filename"])
    if "revision" in model:
        kwargs["revision"] = model["revision"]
    cached = hf_hub_download(**kwargs)
    shutil.copy(cached, dest)
    print(f"  saved → {dest}")


# ---------------------------------------------------------------------------
# Body composite — preserve upper or lower body from original after generation
# ---------------------------------------------------------------------------
def composite_body(result: Image.Image, original_arr, garment_type: str) -> Image.Image:
    """
    Blend the un-swapped body region from the original back onto the result
    using a feathered horizontal mask to avoid hard seams.
    """
    import numpy as np
    if garment_type == "full":
        return result

    rw, rh = result.size
    original = Image.fromarray(original_arr).resize((rw, rh), Image.LANCZOS)

    # Waist is roughly at 57% from top for a standing portrait
    split = 0.57 if garment_type == "upper" else 0.43
    split_px = int(rh * split)
    feather  = int(rh * 0.08)          # smooth over 8% of height

    mask = np.zeros((rh, rw), dtype=np.float32)
    if garment_type == "upper":
        # White (use result) above waist, black (use original) below
        mask[:max(0, split_px - feather)] = 1.0
        for i in range(feather * 2):
            y = split_px - feather + i
            if 0 <= y < rh:
                mask[y] = 1.0 - (i / (feather * 2))
    else:
        # White (use result) below waist, black (use original) above
        mask[min(rh, split_px + feather):] = 1.0
        for i in range(feather * 2):
            y = split_px - feather + i
            if 0 <= y < rh:
                mask[y] = i / (feather * 2)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return Image.composite(result, original, mask_img)


# ---------------------------------------------------------------------------
# Product photo detector — if garment is already on plain background, skip SAM3
# ---------------------------------------------------------------------------
def is_product_photo(img_pil: Image.Image, bg_threshold: float = 0.80) -> bool:
    """
    Returns True if the garment image has a plain, uniform background
    (white, beige, cream, grey — i.e. a product/hanger shot, not a lifestyle photo).
    Uses average brightness + low colour variance rather than per-channel thresholds,
    so beige/cream backgrounds are correctly identified.
    """
    import numpy as np
    arr = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    m_h, m_w = max(1, h // 8), max(1, w // 8)
    corners = [
        arr[:m_h, :m_w], arr[:m_h, -m_w:],
        arr[-m_h:, :m_w], arr[-m_h:, -m_w:],
    ]
    pixels = np.concatenate([c.reshape(-1, 3) for c in corners]).astype(float)

    # Plain background: high average brightness AND low per-pixel colour variance
    avg_brightness = pixels.mean(axis=1)        # mean of R,G,B per pixel
    channel_std    = pixels.std(axis=1)         # colour variance per pixel (low = neutral tone)

    bright_ratio  = float((avg_brightness > 175).mean())
    uniform_ratio = float((channel_std < 30).mean())

    result = bright_ratio > bg_threshold and uniform_ratio > 0.70
    print(f"Product photo: {result} (bright={bright_ratio:.2f}, uniform={uniform_ratio:.2f})")
    return result


# ---------------------------------------------------------------------------
# Garment type classifier (CLIP zero-shot, runs on CPU)
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("Loading CLIP classifier...")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor

_GARMENT_LABELS = [
    "upper body clothing: shirt, top, blouse, jacket, sweater, hoodie",
    "lower body clothing: pants, jeans, shorts, skirt, trousers",
    "full body clothing: dress, jumpsuit, romper, overall, bodysuit",
]
_GARMENT_TYPES  = ["upper", "lower", "full"]

_PROMPTS = {
    "upper": (
        "start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "Do NOT change the pants, trousers, shorts, skirt, shoes or the accessories — keep the entire lower body from Picture 1 pixel-perfect. "
        "Remove only the shirt, top, or jacket from Picture 1 and replace it with the exact garment shown in Picture 2, "
        "strictly preserving the color, graphic, pattern, and material of the garment in Picture 2. "
        "Do not add or change any lower body clothing. "
        "Match the pose from Picture 1, high quality, sharp details, 4k. "
        "Preserve the face, hair and expression from Picture 1."
    ),
    "lower": (
        "start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "Keep the top/shirt/jacket and the shoes/feet/accessories from Picture 1 exactly as they are. "
        "Remove only the pants/skirt/shorts from Picture 1 and replace them with the garment from Picture 2, "
        "strictly preserving its (garment from Picture 2) color, material and design. "
        "Match the pose from Picture 1, high quality, sharp details, 4k. "
        "Preserve the face and expression from Picture 1."
    ),
    "full": (
        "start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "Remove the entire outfit from Picture 1 and replace it completely with the garment from Picture 2, "
        "strictly preserving its (garment from Picture 2) color, material and design. "
        "Match the pose from Picture 1, high quality, sharp details, 4k. "
        "Preserve the face and expression from Picture 1."
    ),
}

def classify_garment(img_pil: Image.Image) -> str:
    """Returns 'upper', 'lower', or 'full'."""
    model, processor = _load_clip()
    inputs = processor(text=_GARMENT_LABELS, images=img_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    idx   = probs.argmax().item()
    label = _GARMENT_TYPES[idx]
    print(f"Garment classified as: {label} (scores: {probs.tolist()})")
    return label


# ---------------------------------------------------------------------------
# Setup — split into two parts:
#   setup_env()   : clone repos, install packages, download models (no GPU needed)
#   start_comfyui(): launch ComfyUI subprocess (must run inside @spaces.GPU)
# ---------------------------------------------------------------------------
def setup_env():
    # 1. Clone ComfyUI
    if not Path(f"{COMFYUI_DIR}/main.py").exists():
        print("=== Cloning ComfyUI ===")
        run(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFYUI_DIR}")
    print("=== Installing ComfyUI requirements ===")
    run(f"pip install -r {COMFYUI_DIR}/requirements.txt -q --break-system-packages")

    # 2. Custom nodes
    print("=== Installing custom nodes ===")
    os.makedirs(COMFYUI_CUSTOM_NODES, exist_ok=True)
    for name, url in CUSTOM_NODES.items():
        node_dir = Path(f"{COMFYUI_CUSTOM_NODES}/{name}")
        if not node_dir.exists():
            print(f"  cloning {name}")
            run(f"git clone --depth 1 {url} {node_dir}")
        req = node_dir / "requirements.txt"
        if req.exists():
            run(f"pip install -r {req} -q --break-system-packages")

    # 3. Models
    print("=== Downloading models ===")
    for model in MODELS:
        download_model(model)

    # 4. Config
    config_path = Path(f"{COMFYUI_DIR}/user/__manager/config.ini")
    if config_path.exists():
        content = config_path.read_text()
        content = content.replace("network_mode = public", "network_mode = personal_cloud")
        content = content.replace("security_level = strict", "security_level = normal")
        config_path.write_text(content)


_comfyui_started = False

def start_comfyui():
    """Start ComfyUI subprocess. Must be called from within a @spaces.GPU context."""
    global _comfyui_started
    if _comfyui_started:
        return
    print("=== Starting ComfyUI ===")
    os.makedirs(COMFYUI_INPUT, exist_ok=True)
    os.makedirs(COMFYUI_OUTPUT, exist_ok=True)

    proc = subprocess.Popen(
        f"python {COMFYUI_DIR}/main.py --listen --port 8188",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def _stream(p):
        for line in p.stdout:
            print(line.decode(errors="replace"), end="")
    threading.Thread(target=_stream, args=(proc,), daemon=True).start()

    print("Waiting for ComfyUI...")
    for _ in range(60):
        try:
            if requests.get(f"{COMFYUI_URL}/system_stats", timeout=3).status_code == 200:
                print("ComfyUI ready!")
                _comfyui_started = True
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI failed to start within 3 minutes")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@spaces.GPU(duration=180)
def generate(target_img, clothing_img, progress=gr.Progress(track_tqdm=True)):
    if target_img is None or clothing_img is None:
        raise gr.Error("Please upload both images before generating.")

    start_comfyui()

    # Clear ComfyUI node output cache so previous results don't bleed into this run
    try:
        requests.post(f"{COMFYUI_URL}/free", json={"unload_models": False, "free_memory": True})
    except Exception:
        pass

    # Save user images to ComfyUI input folder (cap at 1024px to save GPU time)
    def _prep(arr):
        import numpy as np
        img = Image.fromarray(arr)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        # Add 1-pixel invisible noise so content hash is unique every run,
        # preventing ComfyUI from serving cached node outputs.
        px = np.array(img)
        px[0, 0, 0] = (int(px[0, 0, 0]) + 1) % 256
        return Image.fromarray(px)

    uid = uuid.uuid4().hex[:8]
    target_name   = f"target_{uid}.png"
    clothing_name = f"clothing_{uid}.png"
    clothing_pil  = _prep(clothing_img)
    _prep(target_img).save(f"{COMFYUI_INPUT}/{target_name}")
    clothing_pil.save(f"{COMFYUI_INPUT}/{clothing_name}")

    # Classify garment type and check if it's a product photo
    garment_type   = classify_garment(clothing_pil)
    product_photo  = is_product_photo(clothing_pil)

    # Load workflow and inject filenames, seed, and dynamic prompt
    with open("workflow_api.json") as f:
        workflow = json.load(f)
    workflow["76"]["inputs"]["image"]       = target_name
    workflow["81"]["inputs"]["image"]       = clothing_name
    workflow["104"]["inputs"]["noise_seed"] = int(time.time() * 1000) % (2 ** 32)
    workflow["107"]["inputs"]["text"]       = _PROMPTS[garment_type]

    # If garment is already on a plain background, bypass SAM3 + PersonMaskUltra
    # and feed the raw image directly to the scale node
    if product_photo:
        print("Bypassing SAM3 segmentation — using raw clothing image")
        workflow["110"]["inputs"]["image"] = ["81", 0]

    # Submit to ComfyUI
    progress(0.05, desc="Submitting to ComfyUI...")
    client_id = uuid.uuid4().hex
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
    )
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    # Poll /history until done
    progress(0.1, desc="Generating — this takes 1–2 minutes...")
    started = time.time()
    while True:
        history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or entry.get("error"):
                raise gr.Error(f"Generation failed: {entry.get('error', 'unknown error')}")
            break
        elapsed = int(time.time() - started)
        progress(min(0.9, 0.1 + elapsed / 150 * 0.8), desc=f"Generating... ({elapsed}s)")
        time.sleep(3)

    # Retrieve output image
    outputs = history[prompt_id].get("outputs", {})
    for node_output in outputs.values():
        if "images" in node_output:
            img_info = node_output["images"][0]
            img_bytes = requests.get(
                f"{COMFYUI_URL}/view",
                params={
                    "filename": img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type": img_info.get("type", "output"),
                },
            ).content
            progress(1.0, desc="Done!")
            result = Image.open(BytesIO(img_bytes))
            tmp_path = f"/tmp/cloth_swap_{uid}.png"
            result.save(tmp_path)
            type_display = {"upper": "👕 Upper body", "lower": "👖 Lower body", "full": "👗 Full body"}
            label_md = gr.update(visible=True, value=f"*Detected garment type: **{type_display[garment_type]}***")
            return result, gr.update(value=tmp_path, visible=True), label_md

    raise gr.Error("No output image was returned by ComfyUI.")  # noqa: returns (image, file) on success


# ---------------------------------------------------------------------------
# Access code — set the same code somewhere visible in your CV
# ---------------------------------------------------------------------------
ACCESS_CODE = os.getenv("SECRET_CODE", "")

# Pure client-side unlock — no server roundtrip, no loading spinner.
# The code is embedded in JS (same security level as a plain-text textbox).
_gate_js = f"""
function() {{
    const CODE = {repr(ACCESS_CODE)};

    function injectCSS(id, rules) {{
        let el = document.getElementById(id);
        if (!el) {{ el = document.createElement('style'); el.id = id; document.head.appendChild(el); }}
        el.textContent = rules;
    }}

    function tryUnlock() {{
        const codeEl = document.querySelector('#code_input textarea') ||
                       document.querySelector('#code_input input');
        if (!codeEl) return;
        if (codeEl.value.trim() === CODE) {{
            injectCSS('_unlock_css',
                '#gate_col {{ display: none !important; }}' +
                '#app_col  {{ display: block !important; }}' +
                '#error_msg {{ display: none !important; }}'
            );
        }} else {{
            injectCSS('_unlock_css',
                '#error_msg {{ display: block !important; }}'
            );
            const err = document.getElementById('error_msg');
            if (err) {{
                const p = err.querySelector('p') || err;
                p.textContent = 'Incorrect code — check your CV and try again.';
            }}
        }}
    }}

    function wireUp() {{
        const btn    = document.getElementById('unlock_btn');
        const codeEl = document.querySelector('#code_input textarea') ||
                       document.querySelector('#code_input input');
        if (!btn || !codeEl) return false;
        btn.addEventListener('click', tryUnlock);
        codeEl.addEventListener('keydown', (e) => {{
            if (e.key === 'Enter') {{ e.preventDefault(); tryUnlock(); }}
        }});
        return true;
    }}

    let attempts = 0;
    const poll = setInterval(() => {{
        if (wireUp() || ++attempts > 30) clearInterval(poll);
    }}, 200);
}}
"""

CV_FILE = "Sofia_Metelitsa_CV.pdf"  

# ---------------------------------------------------------------------------
# Run setup at module load (no GPU needed). ComfyUI starts inside @spaces.GPU.
# ---------------------------------------------------------------------------
setup_env()

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Virtual Try On", theme=gr.themes.Soft(), js=_gate_js) as demo:

    # ── Gate screen ──────────────────────────────────────────────────────────
    with gr.Column(visible=True, elem_id="gate_col") as gate:
        gr.Markdown(
            """
            *AI-powered outfit swapping built with FLUX.2 Kontext + ComfyUI*

            ---

            ### 🔑 To access this tool, download my CV.
            The access code is inside the CV.
            """
        )
        dl_btn = gr.DownloadButton(
            label="📄 Download CV",
            value=CV_FILE,
            variant="secondary",
            size="lg",
        )

        gr.Markdown("---")
        code_input = gr.Textbox(
            label="Enter access code from CV",
            placeholder="Access code",
            type="text",
            max_lines=1,
            elem_id="code_input",
        )
        unlock_btn = gr.Button("Unlock →", variant="primary", elem_id="unlock_btn")
        error_msg  = gr.Markdown("Incorrect code — check your CV and try again.", visible=False, elem_id="error_msg")

    # ── Main app (hidden until unlocked) ─────────────────────────────────────
    # Build preset image lists
    _people_paths   = sorted(Path("images/people").glob("*.*"))
    _garments_paths = sorted(Path("images/garments").glob("*.*"))

    with gr.Column(visible=False, elem_id="app_col") as main_app:
        gr.Markdown(
            """
            # Virtual Try On
            Pick a **person** and a **garment** from the presets below, or upload your own.
            The AI swaps the outfit while preserving pose, lighting, and expression.
            """
        )
        with gr.Row():
            target_input   = gr.Image(label="Person", type="numpy", height=350)
            clothing_input = gr.Image(label="Selected garment", type="numpy", height=350, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("**People presets — click to select**")
                gr.Examples(
                    examples=[[str(p)] for p in _people_paths],
                    inputs=[target_input],
                    label=None,
                    examples_per_page=8,
                )
            with gr.Column():
                gr.Markdown("**Garment presets — click to select**")
                gr.Examples(
                    examples=[[str(p)] for p in _garments_paths],
                    inputs=[clothing_input],
                    label=None,
                    examples_per_page=12,
                )

        garment_label = gr.Markdown(visible=False)
        btn      = gr.Button("✨ Generate", variant="primary", size="lg")
        output   = gr.Image(label="Result", height=500)
        download = gr.File(label="⬇️ Download result", visible=False)
        gr.Markdown(
            "*Generation takes ~2–3 minutes on this hardware. "
            "Results may vary — if the swap doesn't look right, try hitting Generate again. "
            "Each run uses a different random seed so you may get a better result on the next try.*"
        )

        btn.click(fn=generate, inputs=[target_input, clothing_input], outputs=[output, download, garment_label])

    # Unlock is handled entirely client-side via _gate_js — no server call needed.

# The HF base image ships websockets v13+ which removed the legacy API that
# the installed uvicorn version uses. Redirect uvicorn's WebSocket backend to
# wsproto before demo.launch() triggers the import of uvicorn.protocols.websockets.auto.
import sys as _sys
from types import ModuleType as _ModuleType
_ws_auto = _ModuleType("uvicorn.protocols.websockets.auto")
_sys.modules["uvicorn.protocols.websockets.auto"] = _ws_auto
try:
    from uvicorn.protocols.websockets.wsproto_impl import WSProtocol as _WSP
    _ws_auto.AutoWebSocketsProtocol = _WSP
    print("uvicorn → wsproto WebSocket backend active")
except Exception as _e:
    print(f"WARNING: wsproto backend setup failed: {_e}")

# When server_name="0.0.0.0", gradio constructs local_url as "http://0.0.0.0:7860/"
# which is a bind address and can't be used as a connection target — url_ok fails.
# The server IS running; patch url_ok so gradio doesn't block on this false negative.
try:
    import gradio.networking as _gnet
    _gnet.url_ok = lambda url: True
except Exception as _e:
    print(f"url_ok patch failed: {_e}")

# Starlette 0.36+ changed TemplateResponse(name, context) → TemplateResponse(request, name, context).
# Gradio 4.44.0 still uses the old signature, so "index.html" ends up as `request`
# and the context dict ends up as `name`. Patch to restore the old behaviour.
try:
    import starlette.templating as _st
    _orig_TR = _st.Jinja2Templates.TemplateResponse
    def _compat_TR(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            # Old-style call: TemplateResponse("template.html", context_dict, ...)
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", {})
            request = context.get("request")
            return _orig_TR(self, request, name, context, **kwargs)
        return _orig_TR(self, *args, **kwargs)
    _st.Jinja2Templates.TemplateResponse = _compat_TR
except Exception as _e:
    print(f"starlette TemplateResponse patch failed: {_e}")

# Jinja2 3.1.4 bug: LRUCache uses unhashable dict as cache key when globals is
# non-empty, causing TemplateResponse to crash. Convert TypeError → KeyError so
# templates are loaded fresh when the key can't be cached.
try:
    import jinja2.utils as _jutils
    _LRU = _jutils.LRUCache
    _orig_gi = _LRU.__getitem__
    _orig_si = _LRU.__setitem__
    def _safe_gi(self, key):
        try:
            return _orig_gi(self, key)
        except TypeError:
            raise KeyError(key)
    def _safe_si(self, key, value):
        try:
            _orig_si(self, key, value)
        except TypeError:
            pass
    _LRU.__getitem__ = _safe_gi
    _LRU.__setitem__ = _safe_si
except Exception as _e:
    print(f"jinja2 LRUCache patch failed: {_e}")

# gradio_client bug: _json_schema_to_python_type() can't handle bool schemas
# (e.g. additionalProperties: true). Patch the internal recursive function so
# any non-dict schema is treated as "Any" instead of raising APIInfoParseError.
try:
    import gradio_client.utils as _gcu
    _orig_j2p = _gcu._json_schema_to_python_type
    def _safe_j2p(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_j2p(schema, defs)
    _gcu._json_schema_to_python_type = _safe_j2p
except Exception as _e:
    print(f"gradio_client patch failed: {_e}")

demo.launch(server_name="0.0.0.0", show_error=True, show_api=False)

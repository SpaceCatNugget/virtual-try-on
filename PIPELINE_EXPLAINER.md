# Virtual Try-On — Pipeline Explainer

## The One-Sentence Version

> A user picks a person photo and a garment; the system extracts the garment, classifies what body part it covers, and feeds both images into a distilled image-generation model that re-dresses the person while keeping their pose, face, and background intact.

---

## The Full Pipeline

### 1. User Input
The user selects a **person image** (upload or preset) and a **garment image** (presets only — prevents misuse). Both are sent to the backend for processing.

---

### 2. Garment Classification (CLIP)
Before anything generates, we run a **CLIP zero-shot classifier** on the garment image.

CLIP encodes the image and compares it against three text descriptions:
- "upper body garment (shirt, jacket, top)"
- "lower body garment (pants, skirt, shorts)"
- "full body garment (dress, jumpsuit, romper)"

The winner determines two things: which **text prompt** to use, and implicitly, which body parts the model should touch. If it's upper body, the prompt says *"keep the pants, only replace the shirt."* If full body, it says *"replace the entire outfit."*

---

### 3. Product Photo Detection
Before running heavy segmentation, we check: is the garment image a flat-lay product photo (clean white/plain background), or is it worn by a person?

The check samples the four **corner regions** of the image. If average brightness is above 175 and per-pixel channel standard deviation is below 30, it's a plain background — we skip segmentation and use the garment image directly. If a person is wearing it, we need to extract just the clothing first.

This matters because SAM3 looks for a person to segment clothing off of — if there's no person, it either fails or returns garbage.

---

### 4. Garment Segmentation (SAM3 + PersonMaskUltra V2)
For non-product photos, we run **SAM3** (Segment Anything Model 3) with the text prompt `"clothing, garment, outfit, shirt, pants, dress, jacket"` to locate the clothing region.

The mask is then refined by **PersonMaskUltra V2**, which produces a clean clothing mask using VITMatte matting (handles fine edges like shirt collars). The background is filled with white, and only the garment region is passed forward.

---

### 5. Image Encoding (VAE)
Both the person image and the prepared garment image are encoded into **latent space** using the FLUX.2 VAE. This compressed representation is what the model actually works with — not the raw pixels.

---

### 6. Reference Conditioning (FLUX.2 Kontext / ReferenceLatent)
This is the core mechanism. FLUX.2 Klein supports **ReferenceLatent** — a way to condition generation directly on image latents, not just text.

The workflow chains three ReferenceLatent nodes:
- The **person image latent** informs the structure, pose, lighting, and background
- The **garment image latent** informs texture, color, pattern, and fabric
- A **text prompt** (encoded via the Qwen 3 4B CLIP model) provides semantic guidance about what to change and what to preserve

Together, these three conditioning signals tell the model: *"Generate an image that looks like this person, but wearing this garment."*

---

### 7. Sampling (FLUX.2 Klein 4B, Q8 GGUF)
The denoiser runs for **8 steps** using the Euler sampler with **CFG = 1** (no classifier-free guidance — distilled models don't need it).

The model itself is FLUX.2 Klein 4B: a distilled, 4-billion-parameter rectified flow transformer. "Distilled" means it was trained to match a larger teacher model in very few steps. "Rectified flow" means the denoising path from noise to image is straight rather than curved, making fewer steps sufficient.

It's loaded in **Q8 GGUF** format: 8-bit quantization that cuts VRAM usage roughly in half vs fp16 with negligible quality loss.

---

### 8. Decode and Return
The output latent is decoded back to pixels by the VAE. ComfyUI saves the image to disk. The Python backend polls ComfyUI's `/history` endpoint every second until the job finishes, then fetches the image via `/view` and returns it to the Gradio UI.

To prevent stale cached outputs between runs, two things happen before each generation: ComfyUI's node cache is cleared via `/free`, and a single pixel in the input image is flipped by 1 value so the content hash is unique.

---

### Infrastructure
- **Frontend**: Gradio 4.44.0 on HuggingFace Spaces
- **GPU**: NVIDIA T4 Small (16GB VRAM), persistent — no per-user quota
- **Inference engine**: ComfyUI runs as a subprocess, exposed over a local HTTP API
- **Access gate**: JavaScript-only code check — no server call, instant toggle

---

## Hard Questions They Might Ask

---

**Q: Why use ReferenceLatent instead of inpainting?**

A: We tried inpainting — segmenting the person's clothing region and compositing a warped garment into it. The results looked bad: visible seams, texture mismatch at the edges, and no natural adaptation to the person's pose or body shape. ReferenceLatent is generative: the model synthesizes the dressed result from scratch, guided by both images simultaneously. It naturally handles wrinkles, shadows, and pose adaptation because it's not pasting — it's generating.

---

**Q: What exactly is ReferenceLatent / FLUX.2 Kontext and how does it differ from regular image conditioning?**

A: Standard diffusion conditioning works through cross-attention on text embeddings — the model attends to token vectors at each denoising step. ReferenceLatent is FLUX.2 Kontext's mechanism for conditioning on images: it encodes a reference image into latent space and concatenates it with the latent being denoised (or injects it via attention). This means the model can directly copy visual features — exact fabric patterns, colors, textures — from the reference, which text descriptions can never fully capture. It's closer to how Stable Diffusion's IP-Adapter works, but native to the FLUX.2 architecture.

---

**Q: Why ComfyUI instead of running inference directly in Python?**

A: ComfyUI gives a pre-built, composable graph of nodes — GGUF loader, SAM3, PersonMaskUltra, ReferenceLatent, VAE encode/decode — that would each take significant effort to wire together from scratch. It also handles model memory management and GPU scheduling. We treat it as a black-box inference server: app.py submits a JSON workflow via HTTP and polls for results. The tradeoff is we can't easily introspect intermediate tensors, but for a demo-scale project the productivity gain is worth it.

---

**Q: Why Q8 and not Q4 quantization?**

A: Q8 (8-bit) has virtually no perceptual quality loss compared to fp16. Q4 (4-bit) is more aggressive — it halves VRAM again but introduces visible artifacts especially in textures and fine details, which matter a lot for clothing patterns. For a try-on demo where garment fidelity is the whole point, Q4 degradation is noticeable. Q8 gives us the VRAM savings we need to fit on a T4 while keeping quality acceptable.

---

**Q: Why 8 steps instead of 4 (the model's distillation target)?**

A: FLUX.2 Klein was distilled to converge in 4 steps, so 4 is theoretically sufficient. In practice, complex garment textures (stripes, patterns, fine weave) benefit from the extra passes — 4 steps sometimes produces blurry or smeared fabric details. 8 steps roughly doubles generation time but meaningfully improves texture fidelity. It's a quality/speed tradeoff tuned empirically for the use case.

---

**Q: What is rectified flow and why does it matter?**

A: Standard diffusion models learn to reverse a noise process along a curved, non-linear path — the score function at each noise level is different, which is why you need many steps. Rectified flow straightens this path: the model learns a vector field that maps noise to data along nearly straight trajectories. Straight paths mean fewer integration steps, more predictable dynamics, and easier distillation. FLUX models are flow matching models — the "diffusion" naming is a bit of a misnomer.

---

**Q: What are the main failure modes?**

A: Three main ones. First, **garment type misclassification** — CLIP occasionally misidentifies a cropped garment, and the wrong prompt means the wrong body part gets changed. Second, **SAM3 over-segmentation** — if the person in the garment photo has skin or accessories near the clothing, the mask bleeds into them, and the garment reference includes noise. Third, **model hallucination** — the generator sometimes makes up garment details or drifts from the reference, especially for complex prints. The cache-busting and re-run option address the third; the product photo bypass addresses the second.

---

**Q: How would you scale this beyond a single T4?**

A: A few layers. Horizontally: run multiple Space replicas behind HuggingFace's load balancer, each with its own T4 and ComfyUI instance. Vertically: upgrade to A100 (80GB) to run fp16 instead of Q8 and reduce generation to ~30 seconds. Architecturally: decouple the CPU work (SAM3 segmentation, CLIP classification) from the GPU generation into separate services, so the expensive GPU isn't blocked during pre-processing. For very high throughput, batch requests and use async queuing (Celery or similar) rather than Gradio's built-in queue.

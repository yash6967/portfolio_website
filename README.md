# Yash Sharma — Inference Engineer Portfolio Website
*(Design inspired by [Lemma](https://www.uselemma.ai/))*

A high-performance, developer-focused portfolio website tailored for **Inference Engineering**, model serving, quantization, and systems-level ML. 

Designed with the aesthetic identity of **Lemma (uselemma.ai)**:
- **Lemma Signature Palette:** Deep indigo (`#1D1956`), rich midnight purple (`#090814`), and electric violet accents (`#755CFE`).
- **Mathematical Hairline Architecture:** Crisp 0.5px hairline borders, sharp technical cards (`radius: 0px`), coordinate markers, and catalog indices (`[SPEC. 01]`, `[INF. 01]`).
- **Interactive Trace & Profiler Terminal:** Live multi-tab terminal showing real inference traces (`vLLM continuous batching`, `AutoAWQ INT4 calibration`, `Triton gRPC`, and `FlashAttention-2 roofline analysis`).
- **Zero Build Step:** 100% vanilla HTML, CSS, and JavaScript. Lightning-fast deployments on **Vercel**.

---

## ⚡ Key Highlights

1. **Inference-First Layout:** Projects and benchmark metrics are placed directly under the hero section.
2. **Interactive Benchmark Terminal:** Allows visitors to switch between actual inference profiling logs.
3. **Trace-Style Project Cards:** Styled with audit boxes, trace IDs, TTFT metrics, and memory savings.
4. **Systems Matrix:** 2x2 mathematical grid displaying your inference stack (vLLM, TensorRT-LLM, AWQ, GPTQ, Triton, CUDA, PyTorch Profiler).
5. **Lemma Light & Dark Modes:** Toggle between Lemma's signature dark midnight-indigo mode and clean daylight violet theme.

---

## 📁 File Structure

```
yash-portfolio/
├── index.html        # Main semantic markup & Lemma UI components
├── style.css         # Lemma design system, tokens, hairline borders, gradients
├── script.js         # Interactive trace terminal simulator, filter, theme toggle
├── vercel.json       # Vercel deployment headers & clean URLs
└── README.md         # Documentation & deployment guide
```

---

## 💻 Local Preview

Run in your terminal inside this directory:
```powershell
python -m http.server 3000
```
Then open `http://localhost:3000` in your browser.

---

## 🚀 How to Host on Vercel

### Option A: Via GitHub (Recommended)
1. Push this folder to your GitHub:
   ```bash
   git init
   git add .
   git commit -m "Inference Engineer portfolio with Lemma design language"
   git branch -M main
   git remote add origin https://github.com/yash6967/portfolio.git
   git push -u origin main
   ```
2. Log into [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import `portfolio`, keep preset as **Other**, and click **Deploy**.

### Option B: Via Vercel CLI
```bash
npx vercel
```

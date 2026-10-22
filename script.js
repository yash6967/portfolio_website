document.addEventListener('DOMContentLoaded', () => {
  // 1. Dynamic Year in Footer
  const yearSpan = document.getElementById('year');
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear();
  }

  // 2. Dark / Light Theme Toggle (Lemma Theme System)
  const themeToggle = document.getElementById('theme-toggle');
  const htmlRoot = document.documentElement;

  const savedTheme = localStorage.getItem('theme');
  const systemPrefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;

  if (savedTheme) {
    htmlRoot.setAttribute('data-theme', savedTheme);
  } else if (systemPrefersLight) {
    htmlRoot.setAttribute('data-theme', 'light');
  } else {
    htmlRoot.setAttribute('data-theme', 'dark');
  }

  themeToggle?.addEventListener('click', () => {
    const currentTheme = htmlRoot.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    htmlRoot.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  });

  // 3. Mobile Navigation Menu Toggle
  const mobileToggle = document.getElementById('mobile-toggle');
  const navLinks = document.getElementById('nav-links');

  mobileToggle?.addEventListener('click', () => {
    navLinks?.classList.toggle('open');
  });

  navLinks?.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
    });
  });

  // 4. Interactive Lemma-Style Terminal Simulator
  const terminalData = {
    vllm: `
<div class="term-line"><span class="term-prompt">›</span><span class="term-cmd">vllm serve meta-llama/Meta-Llama-3-8B-Instruct --gpu-memory-utilization 0.92 --max-model-len 8192</span></div>
<div class="term-line" style="margin-top: 10px;"><span class="term-prompt">●</span><span class="term-cmd">engine_init · <span class="term-code">PagedAttentionBackend(KV_block_size=16)</span></span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Allocating 2,496 GPU blocks (~39,936 tokens of KV cache)</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Continuous batching enabled (iteration-level scheduling)</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ Engine ready — serving on http://0.0.0.0:8000/v1</span></div>

<div class="term-line" style="margin-top: 14px;"><span class="term-prompt">›</span><span class="term-cmd">benchmark-serving --concurrency 64 --input-len 1024 --output-len 256</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-code">Time To First Token (TTFT):</span> <span class="term-success">14.2ms (p50) · 18.6ms (p99) [-42% vs eager]</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-code">Inter-Token Latency (ITL):</span> <span class="term-meta">4.8ms/token · Total Throughput: 114 tok/sec</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ Benchmark completed with 0 dropped requests</span> <span class="term-cursor"></span></div>
    `,
    awq: `
<div class="term-line"><span class="term-prompt">›</span><span class="term-cmd">python -m awq.entry --model_path meta-llama/Llama-3-8B --w_bit 4 --q_group_size 128 --run_awq</span></div>
<div class="term-line" style="margin-top: 10px;"><span class="term-prompt">●</span><span class="term-cmd">quantization · <span class="term-code">AutoAWQ::ActivationAwareQuantizer</span></span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Collecting calibration activations from Pile dataset (128 samples)</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Searching channel-wise protection scales (salient weight preservation)</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ Generated AWQ 4-bit group-128 quantization checkpoint</span></div>

<div class="term-line" style="margin-top: 14px;"><span class="term-prompt">›</span><span class="term-cmd">eval-perplexity --dataset wikitext-2 --split test</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-code">FP16 Baseline Perplexity:</span> <span class="term-meta">6.14 (VRAM: 15.9 GB)</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-code">AWQ INT4 Quantized:</span> <span class="term-success">6.23 [Δ +0.09] (VRAM: 5.8 GB -63% cut)</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ Verified memory saving: fits inside single RTX 4070 (12GB VRAM)</span> <span class="term-cursor"></span></div>
    `,
    triton: `
<div class="term-line"><span class="term-prompt">›</span><span class="term-cmd">tritonserver --model-repository=/models --strict-model-config=false --log-verbose=1</span></div>
<div class="term-line" style="margin-top: 10px;"><span class="term-prompt">●</span><span class="term-cmd">model_loader · <span class="term-code">ONNXRuntime_Backend::bge-large-en-v1.5</span></span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Dynamic batch scheduler initialized: max_queue_delay_microseconds: 5000</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Instance group: 2 execution instances per GPU core</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ gRPC Inference Service listening on 0.0.0.0:8001</span></div>

<div class="term-line" style="margin-top: 14px;"><span class="term-prompt">›</span><span class="term-cmd">perf_analyzer -m bge-large -u 0.0.0.0:8001 -i grpc --concurrency-range 10:80:10</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-code">Max Concurrency:</span> <span class="term-meta">80 requests · Inferences per second: 1,840 inf/sec</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-code">Latency Percentiles:</span> <span class="term-success">p50: 8.2ms · p95: 14.1ms · p99: 17.8ms (&lt;18ms SLA)</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ SLA compliance: 100% within acceptable bounds</span> <span class="term-cursor"></span></div>
    `,
    flash: `
<div class="term-line"><span class="term-prompt">›</span><span class="term-cmd">nsys profile --trace=cuda,nvtx,cublas -o profile_attn_step python run_attention.py</span></div>
<div class="term-line" style="margin-top: 10px;"><span class="term-prompt">●</span><span class="term-cmd">profiler · <span class="term-code">FlashAttention-2 Roofline Breakdown</span></span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Roofline Target: NVIDIA Ada Lovelace SM_89 (1,008 GB/s HBM bandwidth)</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">Standard Attention: DRAM Read/Write volume = 14.2 GB/layer (Memory bound)</span></div>
<div class="term-line"><span class="term-branch">├</span><span class="term-meta">FlashAttention-2: Tiled SRAM Read/Write volume = 1.9 GB/layer [-86% memory traffic]</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ Kernel execution speedup: 2.14x faster execution time</span></div>

<div class="term-line" style="margin-top: 14px;"><span class="term-prompt">›</span><span class="term-cmd">nvtx-verify --status</span></div>
<div class="term-line"><span class="term-branch">└</span><span class="term-success">✓ Zero CUDA out-of-memory errors across 32k context sequence windows</span> <span class="term-cursor"></span></div>
    `
  };

  const termNav = document.getElementById('terminal-nav');
  const termContent = document.getElementById('terminal-content');

  function renderTerminal(tabName) {
    if (termContent && terminalData[tabName]) {
      termContent.innerHTML = terminalData[tabName];
    }
  }

  // Initial render
  renderTerminal('vllm');

  termNav?.querySelectorAll('.term-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      termNav.querySelectorAll('.term-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const tabName = tab.getAttribute('data-tab');
      if (tabName) renderTerminal(tabName);
    });
  });

  // 5. Project Filtering System
  const filterBtns = document.querySelectorAll('.filter-btn');
  const projectCards = document.querySelectorAll('.project-card');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filterValue = btn.getAttribute('data-filter');

      projectCards.forEach(card => {
        const category = card.getAttribute('data-category');
        
        if (filterValue === 'all' || category === filterValue) {
          card.style.display = 'flex';
          setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
          }, 10);
        } else {
          card.style.opacity = '0';
          card.style.transform = 'translateY(10px)';
          setTimeout(() => {
            card.style.display = 'none';
          }, 150);
        }
      });
    });
  });

  // 6. ScrollSpy - Active Navbar Link on Scroll
  const sections = document.querySelectorAll('section[id]');
  const navItems = document.querySelectorAll('.nav-link');

  const observerOptions = {
    root: null,
    rootMargin: '-25% 0px -70% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const currentId = entry.target.getAttribute('id');
        navItems.forEach(item => {
          if (item.getAttribute('href') === `#${currentId}`) {
            item.classList.add('active');
          } else {
            item.classList.remove('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));

  // 7. Copy Email to Clipboard
  const emailCard = document.getElementById('email-copy-card');
  const emailText = document.getElementById('email-address')?.textContent?.trim() || '';
  const copyBtnText = document.getElementById('copy-btn-text');

  emailCard?.addEventListener('click', async () => {
    try {
      if (navigator.clipboard && emailText) {
        await navigator.clipboard.writeText(emailText);
        if (copyBtnText) {
          const original = copyBtnText.textContent;
          copyBtnText.textContent = 'COPIED! ✓';
          copyBtnText.style.color = '#3fc155';
          setTimeout(() => {
            copyBtnText.textContent = original;
            copyBtnText.style.color = '';
          }, 2000);
        }
      }
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  });

  // 8. Contact Form Handling
  const contactForm = document.getElementById('contact-form');
  const formStatus = document.getElementById('form-status');

  contactForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    if (formStatus) {
      formStatus.className = 'form-status success';
      formStatus.textContent = '✓ TRANSMISSION SENT. ACKNOWLEDGED.';
      contactForm.reset();

      setTimeout(() => {
        formStatus.textContent = '';
      }, 5000);
    }
  });
});

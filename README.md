# Anupam Rawat — ML Systems & Applied AI

**IIT Bombay** EE (B.Tech + M.Tech) · Dual Minor CSE + CMInDS · **IBM Research** (neural operators / deep RL)

I build research-grade PyTorch systems and ship production AI — from mesh-agnostic neural operators to structure-aware LLM inference and real-time multimodal agents.

<p align="center">
  <a href="https://novicecoderinfinity.github.io/">🌐 Portfolio</a> ·
  <a href="https://www.linkedin.com/in/anupamrawat001/">LinkedIn</a> ·
  <a href="mailto:anupam.rawat.iitb@gmail.com">Email</a>
</p>

---

## Now

| Track | Focus |
| --- | --- |
| **IBM Research** | **COSMOS** — MoE neural operator (NeurIPS 2026 workshop target) · **ARGO** — deep RL orchestrator for hybrid PDE solving (ICLR 2027 target) |
| **PriorityKV** | Structure-protected mixed-precision KV cache for long agent traces (FlashInfer, H200) |
| **Product AI** | **Ghar** (voice multimodal home design) · **ReVival** (circular commerce agents) |

---

## Featured projects

### [PriorityKV](https://github.com/Arush777/Priority_KV) — ML systems / inference
Structure-aware **BF16/INT4** KV compression so tool schemas and instruction hierarchies survive long multi-turn agent traces. Structure keep **0.688** vs FixedHot **0.125** at matched budget; FlashInfer LSE/packed parity on NVIDIA H200.

### [Ghar](https://github.com/NoviceCoderInfinity/ghar) — real-time multimodal
Voice AI interior designer on **Gemini Live API**: describe a home → concepts → narrated room tours → architect Build Pack. [Short demo](https://youtu.be/y6zc4YT8a3E) · [Full demo](https://youtu.be/1Mt2bvL1mV4)

### [ReVival](https://github.com/NoviceCoderInfinity/ReVival) — Amazon HackOn S6
7-agent circular-commerce pipeline (vision grading, disposition, pricing, buyer matching) on FastAPI + AWS Bedrock. [Live demo](https://revival-rose.vercel.app/)

### IBM Research (ongoing)
- **COSMOS** — discretization-agnostic MoE neural operator; compositional PDE benchmark; reconstruction nMSE cut **180×**; beats FNO / U-Net / persistence on 20-step rollout (Wilcoxon \(p \approx 10^{-46}\))
- **ARGO** — PPO-from-scratch RL system composing neural operators + classical solvers; 4×A100 DDP (**10M** steps / **6.9h**)

---

## Stack

`PyTorch` · `Deep RL (PPO/GAE)` · `Neural Operators` · `FlashInfer` · `Transformers / LLMs` · `DDP / torchrun` · `HPC/LSF` · `FastAPI` · `AWS` · `Gemini Live API`

---

## Selected results

- Structure-aware KV keep **0.688** vs FixedHot **0.125** (PriorityKV, H200)
- COSMOS: beats FNO / U-Net / persistence on 20-step rollout (Wilcoxon \(p \approx 10^{-46}\))
- Wyzr Fintech: +**15%** categorisation accuracy, +**33%** reconciliation match rate (production LLM)
- GeoNLI (ISRO / Inter IIT): **82%** captioning · **83.3%** VQA · **64.9%** grounding

---

## Selected competitions

- **Amazon HackOn Season 6** — ReVival (circular commerce)
- **Qualcomm Innovation Challenge** — Reflection Removal (Rank **4**)
- **Kaggle Image Super Resolution** — Rank **4**
- **Inter IIT Tech Meet 14.0** — ISRO Earth-Observation (GeoNLI)

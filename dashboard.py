"""
dashboard.py
────────────
Phase 0 results dashboard — Streamlit.

Run:
    streamlit run dashboard.py

Reads metrics.db in the project root (build with tools/build_metrics_db.py).
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DB_PATH      = PROJECT_ROOT / "metrics.db"

# Canonical display order and colours for models
MODEL_ORDER  = ["LFM2-VL-450M", "SmolVLM-500M", "MiniCPM-V-4.6",
                "FastVLM-0.5B", "FastVLM-0.5B-FP16", "FastVLM-0.5B-iPhone-FP16",
                "Qwen2.5-VL-3B"]
MODEL_COLORS = {
    "LFM2-VL-450M":           "#1f77b4",   # blue
    "SmolVLM-500M":           "#ff7f0e",   # orange
    "MiniCPM-V-4.6":          "#2ca02c",   # green
    "FastVLM-0.5B":           "#d62728",   # red
    "FastVLM-0.5B-FP16":      "#d62728",
    "FastVLM-0.5B-iPhone-FP16": "#d62728",
    "Qwen2.5-VL-3B":          "#9467bd",   # purple
}

# Trimmed / corrected values noted in ADR-0003
TRIMMED_TPS = {
    "MiniCPM-V-4.6": 33.7,   # run 3 outlier removed (44.0 → trimmed mean 33.7)
}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_db() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)

    iphone = pd.read_sql("SELECT * FROM iphone_perf", conn)
    # Apply trimmed TPS corrections
    for model, tps in TRIMMED_TPS.items():
        iphone.loc[iphone["model_key"] == model, "tps_mean"] = tps

    # Each benchmark stores its top-level score under a different metric name:
    #   POPE          → metric='acc'      (already 0-100)
    #   RealWorldQA   → metric='Overall'  (0-1, multiply ×100)
    #   MMBench_DEV_EN→ metric='Overall'  (0-1, multiply ×100)
    quality = pd.read_sql(
        """
        SELECT model_key, benchmark, metric, value FROM mac_quality
        WHERE (benchmark='POPE'           AND metric='acc')
           OR (benchmark='RealWorldQA'    AND metric='Overall')
           OR (benchmark='MMBench_DEV_EN' AND metric='Overall')
        """,
        conn,
    )
    # Normalise all to 0-100 scale
    quality.loc[quality["benchmark"] != "POPE", "value"] = (
        quality.loc[quality["benchmark"] != "POPE", "value"] * 100
    )
    # Pivot: rows=model, cols=benchmark
    quality_pivot = quality.pivot_table(
        index="model_key", columns="benchmark", values="value"
    ).reset_index()
    quality_pivot.columns.name = None
    for col in ["POPE", "RealWorldQA", "MMBench_DEV_EN"]:
        if col not in quality_pivot.columns:
            quality_pivot[col] = None

    # Deduplicate clip scores — keep one row per logical model
    clip = pd.read_sql(
        """
        SELECT model_key, platform, mean_clip_score, std_clip_score, n
        FROM clip_scores
        GROUP BY model_key
        """,
        conn,
    )
    conn.close()
    return iphone, quality_pivot, clip


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VLM Optimization — Phase 0 Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 VLM Optimization — Phase 0 Baselines")
st.caption(
    "Reference measurements for 4 small-edge VLMs on iPhone 16 Pro (A18 Pro). "
    "All numbers are Phase 0 frozen baselines. Phase 1 optimization experiments "
    "must improve on these."
)

if not DB_PATH.exists():
    st.error(
        f"`metrics.db` not found at `{DB_PATH}`. "
        "Run `python tools/build_metrics_db.py` first."
    )
    st.stop()

iphone_df, quality_df, clip_df = load_db()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 iPhone Performance",
    "🎯 Mac Quality (Benchmarks)",
    "🖼️ CLIP-Score",
    "ℹ️ About",
])


# ══ Tab 1 — iPhone Performance ════════════════════════════════════════════════
with tab1:
    st.subheader("iPhone 16 Pro — TTFT & Throughput (Phase 0 baselines)")
    st.caption(
        "Device: iPhone 16 Pro (iPhone17,1, A18 Pro, iOS 26.5). "
        "Prompt: *'Describe this image briefly.'* · maxTokens=64 · "
        "1 warmup + 5 measured runs · 5 images (sample1–5). "
        "MiniCPM-V TPS is trimmed mean (run 3 outlier excluded — see ADR-0003)."
    )

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.markdown("#### Pareto: TTFT vs Peak Memory")
        fig = go.Figure()
        for _, row in iphone_df.iterrows():
            key = row["model_key"]
            color = MODEL_COLORS.get(key, "#888")
            # Error bar for TTFT stddev
            fig.add_trace(go.Scatter(
                x=[row["peak_memory_mb"]],
                y=[row["ttft_ms_mean"]],
                mode="markers+text",
                name=key,
                text=[key],
                textposition="top right",
                textfont=dict(size=11),
                marker=dict(size=14, color=color),
                error_y=dict(
                    type="data",
                    array=[row["ttft_ms_stddev"] or 0],
                    visible=True,
                    color=color,
                ),
                showlegend=True,
            ))
        fig.update_layout(
            xaxis_title="Peak Memory (MB)  ← lower is better",
            yaxis_title="TTFT (ms)  ← lower is better",
            yaxis_type="log",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Decode Throughput (tokens/sec)")
        tps_fig = px.bar(
            iphone_df.sort_values("tps_mean", ascending=False),
            x="model_key", y="tps_mean",
            color="model_key",
            color_discrete_map=MODEL_COLORS,
            labels={"model_key": "Model", "tps_mean": "TPS (higher is better)"},
            text="tps_mean",
            height=320,
        )
        tps_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        tps_fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(tps_fig, use_container_width=True)

    with col_table:
        st.markdown("#### Detailed numbers")
        display = iphone_df[[
            "model_key", "quantization", "ttft_ms_mean", "ttft_ms_stddev",
            "tps_mean", "peak_memory_mb", "on_disk_mb",
        ]].copy()
        display.columns = ["Model", "Quant", "TTFT ms", "±σ", "TPS", "Mem MB", "Disk MB"]
        display = display.sort_values("TTFT ms")
        st.dataframe(
            display.style
                .format({"TTFT ms": "{:.1f}", "±σ": "{:.1f}", "TPS": "{:.1f}",
                         "Mem MB": "{:.0f}", "Disk MB": "{:.0f}"})
                .highlight_min(subset=["TTFT ms", "Mem MB"], color="#c6efce")
                .highlight_max(subset=["TPS"], color="#c6efce"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Key takeaways")
        st.markdown("""
- **LFM2-VL-450M** dominates: fastest TTFT (14 ms), highest TPS (82), lowest memory (279 MB)
- **SmolVLM-500M** is runner-up in latency (20 ms TTFT) at 1.4× more memory
- **MiniCPM-V-4.6** is the heaviest llama.cpp model (970 MB) but matches FastVLM TPS
- **FastVLM-0.5B** (FP16 MLX) is 50× slower TTFT and 8× more memory than LFM2 — FP16 weight loading dominates; quantised MLX would change this picture significantly
        """)


# ══ Tab 2 — Mac Quality ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("Mac mini M4 — Benchmark Accuracy (Task 2.2)")
    st.caption(
        "100-sample slices of POPE, RealWorldQA, MMBench DEV EN. "
        "Exact-match scoring only (no GPT fallback). "
        "Qwen2.5-VL-3B is included as the Phase 2 starting point (3B params, not a Phase 0 target model)."
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("#### Accuracy by benchmark")
        benchmarks = ["POPE", "RealWorldQA", "MMBench_DEV_EN"]
        bench_labels = {"POPE": "POPE", "RealWorldQA": "RealWorldQA",
                        "MMBench_DEV_EN": "MMBench"}

        # Long-form for grouped bar
        rows = []
        for _, r in quality_df.iterrows():
            for b in benchmarks:
                rows.append({
                    "Model": r["model_key"],
                    "Benchmark": bench_labels.get(b, b),
                    "Accuracy %": r.get(b),
                })
        long_df = pd.DataFrame(rows).dropna(subset=["Accuracy %"])

        fig2 = px.bar(
            long_df,
            x="Model", y="Accuracy %",
            color="Benchmark",
            barmode="group",
            height=400,
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
            text="Accuracy %",
        )
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.update_layout(margin=dict(t=10, b=10), yaxis_range=[0, 105])
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.markdown("#### Score table")
        tbl = quality_df[["model_key", "POPE", "RealWorldQA", "MMBench_DEV_EN"]].copy()
        tbl.columns = ["Model", "POPE %", "RealWorldQA %", "MMBench %"]
        tbl = tbl.sort_values("POPE %", ascending=False)
        st.dataframe(
            tbl.style
                .format({"POPE %": "{:.1f}", "RealWorldQA %": "{:.1f}", "MMBench %": "{:.1f}"},
                        na_rep="—")
                .highlight_max(subset=["POPE %", "RealWorldQA %", "MMBench %"],
                               color="#c6efce", axis=0),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Key takeaways")
        st.markdown("""
- **Qwen2.5-VL-3B** leads on POPE (97%) and RealWorldQA (55%) — 3B params, not a Phase 0 edge target
- **MiniCPM-V-4.6** leads MCQ (RealWorldQA 65%, MMBench 79%) among the ≤500M models
- **LFM2-VL-450M** is 2nd on POPE (92%) at 450M params — best accuracy-per-param
- **FastVLM-0.5B** scores lowest on MCQ benchmarks but note: MCQ scores don't capture description richness (see CLIP-Score tab)
        """)

    # ── Benchmark explanations ──────────────────────────────────────────────
    st.divider()
    st.markdown("### About these benchmarks")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🔍 POPE")
        st.markdown("""
**Polling-based Object Probing Evaluation**

Tests whether a model can correctly answer simple yes/no questions about object presence in an image — e.g. *"Is there a cat in this image?"*

**Why it matters:** Hallucination detector. A model that confidently reports objects that aren't there is unsafe to deploy. POPE specifically targets this failure mode.

**Format:** Yes/No questions, exact-match scoring.
**Score range:** 0–100%. Random baseline = 50%. Good models score > 85%.
**Our slice:** 100 samples from the POPE adversarial split.
        """)

    with col_b:
        st.markdown("#### 🌍 RealWorldQA")
        st.markdown("""
**Real-World Visual Question Answering**

Multiple-choice questions grounded in everyday real-world photos — scenes you'd encounter in daily life (traffic, food, signs, objects). Questions require practical visual understanding, not just pattern recognition.

**Why it matters:** Measures whether a model can answer the kinds of questions a real user would ask about a photo on their phone.

**Format:** Multiple-choice (A/B/C/D), exact-match scoring.
**Score range:** 0–100%. Random baseline = 25%. Good models score > 50%.
**Our slice:** 100 samples.
        """)

    with col_c:
        st.markdown("#### 📐 MMBench")
        st.markdown("""
**Multi-Modal Benchmark (DEV EN)**

A broad multi-skill benchmark covering ~20 visual reasoning abilities: attribute recognition, spatial relationships, counting, commonsense reasoning, celebrity recognition, and more.

**Why it matters:** Stress-tests breadth. A model can ace POPE (just yes/no) while failing at spatial reasoning or counting. MMBench reveals these gaps.

**Format:** Multiple-choice, exact-match scoring.
**Score range:** 0–100%. Random baseline = 25%. Good models score > 60%.
**Our slice:** 100 samples from the DEV EN split.
        """)


# ══ Tab 3 — CLIP-Score ════════════════════════════════════════════════════════
with tab3:
    st.subheader("CLIP-Score — Description Quality")
    st.caption(
        "CLIP model: `openai/clip-vit-large-patch14` · "
        "Prompt: *'Describe what you see in this image.'* · 5 images. "
        "Score = 100 × max(0, cos_sim(CLIP_img, CLIP_txt)). "
        "Typical range for good captions: 25–35. "
        "FastVLM scores are from iPhone FP16; others from Mac MPS bfloat16."
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        fig3 = px.bar(
            clip_df.sort_values("mean_clip_score", ascending=False),
            x="model_key", y="mean_clip_score",
            error_y="std_clip_score",
            color="model_key",
            color_discrete_map=MODEL_COLORS,
            labels={"model_key": "Model", "mean_clip_score": "CLIPScore (higher is better)"},
            text="mean_clip_score",
            height=380,
        )
        fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig3.update_layout(showlegend=False, yaxis_range=[0, 35],
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        st.markdown("#### Score table")
        clip_tbl = clip_df[["model_key", "platform", "mean_clip_score", "std_clip_score"]].copy()
        clip_tbl.columns = ["Model", "Platform", "CLIPScore", "±σ"]
        clip_tbl = clip_tbl.sort_values("CLIPScore", ascending=False)
        st.dataframe(
            clip_tbl.style
                .format({"CLIPScore": "{:.2f}", "±σ": "{:.2f}"})
                .highlight_max(subset=["CLIPScore"], color="#c6efce"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Key takeaways")
        st.markdown("""
- All four models cluster tightly (24–28) — within σ overlap
- **MiniCPM-V-4.6** leads narrowly (28.3); **SmolVLM-500M** trails (24.1)
- **FastVLM's** verbose iPhone descriptions score comparably to LFM2 on Mac — its low MCQ scores don't reflect poor description quality
- CLIP-score will be used in Phase 1 as a quality guard: optimizations must not drop below Phase 0 baseline
        """)

    # ── CLIP-Score explanation ──────────────────────────────────────────────
    st.divider()
    st.markdown("### About CLIP-Score")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🧮 How it's computed")
        st.markdown("""
CLIP (Contrastive Language–Image Pretraining) learns a shared embedding space where images and matching text end up close together.

**CLIPScore** = 100 × max(0, cos_sim(image embedding, text embedding))

Both the image and the generated caption are passed through CLIP separately. The cosine similarity between their embeddings — clamped to [0, 1] and scaled to 100 — is the score.

A high score means CLIP's joint image+language model agrees the caption is a good description of the image. No reference caption is needed — it's **reference-free**.

We use `openai/clip-vit-large-patch14`, the highest-quality public CLIP checkpoint.
        """)

    with col_b:
        st.markdown("#### 📏 What the numbers mean")
        st.markdown("""
| Score | Interpretation |
|---:|---|
| < 20 | Poor — caption unrelated to image |
| 20–25 | Weak — generic or partially wrong |
| 25–30 | Good — typical for accurate captions |
| 30–35 | Strong — detailed, specific captions |
| > 35 | Exceptional — rare in practice |

**Our models score 24–28**, which is solidly in the "good" range — they produce accurate descriptions but not highly detailed ones. This is expected given the 64-token output cap used during iPhone measurement.

**Human COCO captions** (the ceiling) typically score ~32–34 on the same metric.
        """)

    with col_c:
        st.markdown("#### 🎯 Why we use it")
        st.markdown("""
POPE and MMBench only test structured tasks (yes/no, multiple choice). They can't tell you whether a model writes a *good description* of an image — which is the core use case for a mobile VLM assistant.

**CLIP-score fills that gap:**

- Measures **open-ended description quality** without needing human-written references
- Complements MCQ benchmarks — FastVLM scores lowest on POPE/MMBench but comparable on CLIP-score, showing its descriptions are semantically accurate even if its MCQ formatting is weak
- **Phase 1 quality guard:** any optimization (quantization, pruning, distillation) that drops CLIP-score below the Phase 0 baseline fails the quality check, even if it improves latency

Limitation: CLIP-score rewards semantic overlap but not factual precision. A caption that correctly names the objects but gets their relationship wrong can still score well.
        """)


# ══ Tab 4 — About ═════════════════════════════════════════════════════════════
with tab4:
    st.subheader("About this dashboard")
    st.markdown(f"""
**Project:** Multi-Agent VLM Optimization System
**Phase:** Phase 0 — Reference Baselines
**Database:** `{DB_PATH.name}` (built by `tools/build_metrics_db.py`)
**Last built:** {DB_PATH.stat().st_mtime if DB_PATH.exists() else 'N/A'}

### Models measured

| Model | HF ID | Params | Phase |
|---|---|---|---|
| LFM2-VL-450M | `LiquidAI/LFM2-VL-450M` | ~450M | Phase 0 baseline |
| SmolVLM-500M | `HuggingFaceTB/SmolVLM-500M-Instruct` | ~500M | Phase 0 baseline |
| MiniCPM-V-4.6 | `openbmb/MiniCPM-V-4.6` | ~1.3B | Phase 0 baseline |
| FastVLM-0.5B | `apple/FastVLM-0.5B` | ~500M | Phase 0 baseline |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | ~3B | Phase 2 starting point |

### Key decisions (ADRs)

- [ADR-0001](docs/decisions/0001-mac-measurement-methodology.md) — Mac measurement methodology
- [ADR-0002](docs/decisions/0002-ios-measurement-methodology.md) — iOS measurement methodology
- [ADR-0003](docs/decisions/0003-iphone-baseline-numbers.md) — iPhone baseline numbers & sanity check
- [ADR-0004](docs/decisions/0004-stage-a-eval-set.md) — Stage A eval set composition

### Eval set

**Stage A** — 100 photos · 50 COCO reference captions · 45 VQA pairs (COCO VQA v2)
Manifest hash: `e2128ae022b3720375d7c866a037b6d8ec4b399ff92cb59e6065ec9fb7f3e29f`

### How to rebuild the database

```bash
python tools/build_metrics_db.py
streamlit run dashboard.py
```
""")

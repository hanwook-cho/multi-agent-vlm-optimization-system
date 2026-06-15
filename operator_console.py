"""
operator_console.py
────────────────────
Operator console (ADR-0013 H2) — the live ops UI: watch and control runs from the
browser. Distinct from dashboard.py (read-only Phase-0/2 metrics); this drives the
system via the run-control flag and reads the shared state (queues, ledger, logs).

Run:
    streamlit run operator_console.py        →  http://localhost:8501

Tabs: Monitor (live, this build) · Setup (run.yaml, H4) · Approvals (H3).
Global chrome: backend indicator + approvals bell; chat dock placeholder (H2b).
"""

from pathlib import Path

import streamlit as st

from services import run_control as rc
from services import console_data as cd
from services import console_chat as cc
from services import approvals as ap

ROOT = Path(__file__).parent

st.set_page_config(page_title="Operator console", page_icon=":wrench:", layout="wide")

# ── Global bar ────────────────────────────────────────────────────────────────
state = rc.get_state()["state"]
_STATE_ICON = {"run": "🟢 running", "pause": "🟡 paused", "stop": "🔴 stopping", "kill": "⛔ killing"}
backend = cd.backend_label(cd.RUN_YAML, dict(__import__("os").environ))
n_pending = len(cd.pending_approvals())

bar = st.columns([3, 2, 2, 2])
bar[0].subheader("Operator console")
bar[1].markdown(f"**state** · {_STATE_ICON.get(state, state)}")
bar[2].markdown(f"**backend** · {'local · Qwen2.5-7B' if backend == 'local' else 'api · frontier'}")
bar[3].markdown(f"**approvals** · :bell: {n_pending}")

# ── Chat dock (H2b) — one strategist session, available on every tab ───────────
if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.markdown("### Chat — strategist")
    st.caption(f"{'local · Qwen2.5-7B' if backend == 'local' else 'api · frontier'} · proposes & explains; gated actions need approval")
    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])
    with st.form("chat_form", clear_on_submit=True):
        msg = st.text_input("ask / steer the strategist", key="chat_in")
        sent = st.form_submit_button("send", width="stretch")
    if sent and msg.strip():
        st.session_state.chat.append({"role": "user", "content": msg.strip()})
        reply = cc.chat_reply(msg.strip(), history=st.session_state.chat[:-1],
                              backend=backend)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()
    st.divider()
    log_path = st.text_input("run log", value=cd.default_log_path(),
                             help="Auto-points at the newest run in artifacts/logs/.")
    rc1, rc2 = st.columns([1, 1])
    auto = rc1.checkbox("auto", value=False, help="Auto-refresh the Monitor")
    every = rc2.selectbox("every", [2, 5, 10, 30], index=1, label_visibility="collapsed")
    if st.button("↻ refresh", width="stretch"):
        st.rerun()

tab_mon, tab_setup, tab_appr = st.tabs(["Monitor", "Setup", "Approvals"])

# ── Monitor ────────────────────────────────────────────────────────────────────
with tab_mon:
    st.markdown("#### Run control")
    c = st.columns(5)
    if c[0].button("⏸ pause", width="stretch", disabled=state == "pause"):
        rc.set_state("pause", "via console"); st.rerun()
    if c[1].button("▶ resume", width="stretch", disabled=state == "run"):
        rc.set_state("run", "via console"); st.rerun()
    if c[2].button("■ stop", width="stretch"):
        rc.set_state("stop", "via console"); st.rerun()
    if c[3].button("⛔ kill", width="stretch"):
        rc.set_state("kill", "via console"); st.rerun()
    if c[4].button("clear", width="stretch"):
        rc.clear(); st.rerun()
    if state != "run":
        st.warning(f"Run control is **{state}** — the loops will react at their next checkpoint. "
                   "Use resume/clear to return to running.")

    _pending = ap.list_pending()
    if _pending:
        top = _pending[0]
        with st.container(border=True):
            st.markdown(f"**:bell: {len(_pending)} approval(s) pending** — {top['kind']}: {top['summary']}")
            ac = st.columns([1, 1, 4])
            if ac[0].button("approve", key=f"mon_appr_{top['id']}", width="stretch"):
                ap.decide(top["id"], "approve", by="console"); st.rerun()
            if ac[1].button("reject", key=f"mon_rej_{top['id']}", width="stretch"):
                ap.decide(top["id"], "reject", by="console"); st.rerun()
            ac[2].caption("same item as the bell and the Approvals tab")

    def _live_monitor():
        st.markdown("#### Current run")
        prog = cd.parse_progress(cd.log_tail(Path(log_path), n=200)) if log_path else None
        m = st.columns(4)
        m[0].metric("stage", prog["stage"] if prog else "—")
        m[1].metric("step", f"{prog['step']}/{prog['total']}" if prog else "—")
        m[2].metric("loss", f"{prog['loss']:.3f}" if prog else "—")
        m[3].metric("queue", cd.queue_len(cd.CONSTRUCTION_QUEUE) + cd.queue_len(cd.EXPERIMENT_QUEUE))

        st.markdown("#### Recent constructed students")
        rows = cd.recent_constructions()
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("Same-path Overall vs the LFM2-VL-450M benchmark: POPE 87.7 · RWQA 0.42 · MMBench 0.74.")
        else:
            st.info("No constructed-student runs in the ledger yet.")

        st.markdown("#### Live log")
        tail = cd.log_tail(Path(log_path), n=24) if log_path else ""
        st.code(tail or f"(no log at {log_path or 'artifacts/logs/'})", language="text")

    # Auto-refresh the live panel without blocking the controls above.
    st.fragment(_live_monitor, run_every=f"{every}s" if auto else None)()

# ── Setup (H4: form writes run.yaml; today shows the config) ────────────────────
with tab_setup:
    st.markdown("#### Run configuration")
    if cd.RUN_YAML.exists():
        st.caption(f"Active intake — {cd.RUN_YAML}")
        st.code(cd.RUN_YAML.read_text(), language="yaml")
    else:
        ex = ROOT / "configs" / "run.example.yaml"
        st.info("No run.yaml yet. Copy the template below to run.yaml and edit (the form is H4).")
        if ex.exists():
            st.code(ex.read_text(), language="yaml")

# ── Approvals (H3) — one queue, three surfaces (bell, Monitor card, here) ───────
with tab_appr:
    st.markdown("#### Pending approvals")
    pend = ap.list_pending()
    if not pend:
        st.info("No pending approvals. Gated decisions appear here, in the bell, and on Monitor.")
    for r in pend:
        with st.container(border=True):
            st.markdown(f"**{r['kind']}** — {r['summary']}")
            if r.get("detail"):
                st.caption(", ".join(f"{k}: {v}" for k, v in r["detail"].items()))
            st.caption(f"id {r['id']} · requested {r['requested_at'][:19]} · by {r['requested_by']}")
            cols = st.columns([1, 1, 4])
            if cols[0].button("approve", key=f"appr_{r['id']}", width="stretch"):
                ap.decide(r["id"], "approve", by="console"); st.rerun()
            if cols[1].button("reject", key=f"rej_{r['id']}", width="stretch"):
                ap.decide(r["id"], "reject", by="console"); st.rerun()

    decided = [r for r in ap.list_all() if r["status"] != "pending"]
    if decided:
        with st.expander(f"history ({len(decided)})"):
            for r in reversed(decided):
                mark = "✓" if r["status"] == "approved" else "✗"
                st.write(f"{mark} [{r['id']}] {r['kind']}: {r['summary']} — {r['status']}"
                         + (f" · {r['note']}" if r.get("note") else ""))

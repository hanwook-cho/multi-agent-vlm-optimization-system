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
from schemas.run_config import RunConfig, load_run_config, save_run_config

ROOT = Path(__file__).parent

st.set_page_config(page_title="Operator console", page_icon=":wrench:", layout="wide")

# ── Global bar ────────────────────────────────────────────────────────────────
state = rc.get_state()["state"]
_STATE_ICON = {"run": "🟢 running", "pause": "🟡 paused", "stop": "🔴 stopping", "kill": "⛔ killing"}
_default_backend = cd.backend_label(cd.RUN_YAML, dict(__import__("os").environ))
backend = st.session_state.get("backend_sel", _default_backend)  # UI selector wins
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
    sel = st.radio("backend", ["local", "api"],
                   index=0 if _default_backend == "local" else 1,
                   key="backend_sel", horizontal=True)

    # Resolve the effective backend + credentials passed to the strategist.
    eff_backend, api_key, base_url, model = "local", None, None, None
    if sel == "local":
        if cd.local_server_up():
            st.caption("🟢 local · Qwen2.5-7B online")
        else:
            st.caption("🔴 local server offline — run scripts/start_strategist_llm.sh")
    else:
        with st.expander("api configuration", expanded=True):
            api_type = st.selectbox("type", ["anthropic", "openai-compatible"], key="api_type")
            eff_backend = "anthropic" if api_type == "anthropic" else "openai_compat"
            model = st.text_input(
                "model", key="api_model",
                value="claude-sonnet-4-5" if api_type == "anthropic" else "") or None
            if api_type == "openai-compatible":
                base_url = st.text_input("base url", key="api_base_url",
                                         placeholder="https://api.example.com/v1") or None
            api_key = st.text_input("api key", type="password", key="api_key",
                                    placeholder="sk-…  (session only, not saved)") or None
            st.caption("Key is held in this session only — never written to disk or logs."
                       + ("  Falls back to ANTHROPIC_API_KEY if left blank."
                          if api_type == "anthropic" else ""))

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])
    with st.form("chat_form", clear_on_submit=True):
        msg = st.text_input("ask / steer the strategist", key="chat_in")
        sent = st.form_submit_button("send", width="stretch")
    if sent and msg.strip():
        st.session_state.chat.append({"role": "user", "content": msg.strip()})
        reply = cc.chat_reply(msg.strip(), history=st.session_state.chat[:-1],
                              backend=eff_backend, api_key=api_key,
                              base_url=base_url, model=model)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()
    st.caption("proposes & explains; gated actions need approval")
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

    st.markdown("#### Launch a run")
    st.caption("Builds the most recent queued spec (or the default) via the same "
               "`construction_loop` entry point as the CLI — results are identical.")
    lc = st.columns([2, 1, 1, 1.4, 1.4])
    lc_smoke = lc[1].checkbox("smoke", value=False, help="Tiny end-to-end build (proves the loop)")
    lc_eval = lc[2].checkbox("eval", value=True, help="Run the same-path floor-adjusted eval after building")
    lc_seed = lc[3].number_input("seed", value=0, step=1, help="Fix for reproducibility; vary to measure variance")
    lc_gate = lc[4].checkbox("require approval", value=False,
                             help="Block the run on the approval queue (appears in the bell / Approvals tab)")
    if lc[0].button("▶ Run next queued spec", width="stretch", disabled=state in ("stop", "kill")):
        try:
            info = cd.launch_construction(smoke=lc_smoke, eval_after=lc_eval,
                                          seed=int(lc_seed), require_approval=lc_gate)
            st.session_state["last_launch"] = info
            st.success(f"launched pid {info['pid']} — progress streams below")
            st.rerun()
        except Exception as exc:
            st.error(f"launch failed: {exc}")
    if "last_launch" in st.session_state:
        st.caption(f"last launch · pid {st.session_state['last_launch']['pid']} · "
                   f"`{st.session_state['last_launch']['cmd']}`")

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

# ── Setup (H4) — form writes run.yaml ───────────────────────────────────────────
_ALL_BENCH = ["POPE", "RealWorldQA", "MMBench_DEV_EN"]
with tab_setup:
    st.markdown("#### Run configuration")
    st.caption(f"Authorizes the run; writes {cd.RUN_YAML.name}. The agent proposes within this scope.")

    # Prefill from run.yaml, else the example template.
    try:
        _src = cd.RUN_YAML if cd.RUN_YAML.exists() else (ROOT / "configs" / "run.example.yaml")
        cur = load_run_config(_src)
    except Exception:
        cur = RunConfig(goal="")

    _hyp_ids = [r["id"] for r in cd.hypothesis_rows()]
    with st.form("setup_form"):
        goal = st.text_area("goal", value=cur.goal, height=70)
        sc = st.columns(3)
        pope = sc[0].number_input("POPE ≥", value=float(cur.success_criteria.get("POPE", 86.0)), step=1.0)
        rwqa = sc[1].number_input("RealWorldQA ≥", value=float(cur.success_criteria.get("RealWorldQA", 0.42)), step=0.01, format="%.2f")
        mmb = sc[2].number_input("MMBench ≥", value=float(cur.success_criteria.get("MMBench_DEV_EN", 0.74)), step=0.01, format="%.2f")
        c2 = st.columns(2)
        device = c2[0].selectbox("target device", ["mac_mini_m4_16gb", "iphone16pro-001"],
                                 index=0 if cur.target_device == "mac_mini_m4_16gb" else 1)
        cbackend = c2[1].selectbox("chat backend (run default)", ["local", "api"],
                                   index=0 if cur.chat_backend == "local" else 1)
        eval_set = st.multiselect("eval set", _ALL_BENCH,
                                  default=[b for b in cur.eval_set if b in _ALL_BENCH] or _ALL_BENCH)
        allowed = st.multiselect("allowed hypotheses (empty = all open)", _hyp_ids or cur.allowed_hypotheses,
                                 default=[h for h in cur.allowed_hypotheses if not _hyp_ids or h in _hyp_ids])
        notes = st.text_input("notes", value=cur.notes or "")
        saved = st.form_submit_button("save run.yaml", width="stretch")

    if saved:
        try:
            crit = {"POPE": pope, "RealWorldQA": rwqa, "MMBench_DEV_EN": mmb}
            cfg = RunConfig(goal=goal, success_criteria={k: v for k, v in crit.items() if k in eval_set},
                            target_device=device, eval_set=eval_set, allowed_hypotheses=allowed,
                            chat_backend=cbackend, notes=notes or None)
            save_run_config(cfg, cd.RUN_YAML)
            st.success(f"saved {cd.RUN_YAML.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"invalid config: {exc}")

    if cd.RUN_YAML.exists():
        with st.expander("current run.yaml"):
            st.code(cd.RUN_YAML.read_text(), language="yaml")

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

    st.markdown("#### Strategy context")
    st.caption("Why the agent is where it is — context for the decisions above.")
    props = cd.latest_proposals()
    if props:
        for p in props:
            st.markdown(f"**{p['hypothesis'] or p['kind']}** · {p['proposed_at'][:19]}")
            if p["rationale"]:
                st.caption(p["rationale"])
    hrows = cd.hypothesis_rows()
    if hrows:
        with st.expander("hypothesis table"):
            st.dataframe(hrows, width="stretch", hide_index=True)

    # ── Research Analyst (Mode B) — read the literature → verified records → the gate ──
    st.markdown("#### Research Analyst (Mode B)")
    st.caption("Searches arXiv for the open problem, extracts hypothesis records, and "
               "verifies them (citation + verbatim quote) before routing to the gate above. "
               "Needs the local LLM server (or the API backend selected in the sidebar).")
    rb = st.columns([2.4, 1, 1.2])
    ra_query = rb[0].text_input("arXiv query", value="efficient small vision-language model token reduction",
                                key="ra_query", label_visibility="collapsed",
                                placeholder="arXiv query…")
    ra_n = rb[1].number_input("papers", value=5, min_value=1, max_value=20, key="ra_n")
    # Backend readiness — a run with no reachable backend extracts nothing.
    ra_local_ok = cd.local_server_up()
    ra_blocked = False
    if sel == "local":
        if ra_local_ok:
            st.caption("🟢 local · Qwen2.5-7B online — extraction will run")
        else:
            ra_blocked = True
            st.caption("🔴 local server offline — run `scripts/start_strategist_llm.sh` first "
                       "(a run now would fetch papers but extract nothing)")
    else:
        st.caption("🔵 api backend — the launched run reads **ANTHROPIC_API_KEY from the "
                   "environment** (the sidebar key is session-only and isn't passed to the subprocess)")
    if rb[2].button("🔍 Run Analyst", width="stretch", disabled=ra_blocked):
        try:
            info = cd.launch_research(query=ra_query, max_papers=int(ra_n),
                                      backend=("api" if sel == "api" else "local"))
            st.success(f"launched pid {info['pid']} — verified records appear below + at the gate")
            st.rerun()
        except Exception as exc:
            st.error(f"launch failed: {exc}")
    hyps = cd.recent_hypotheses()
    if hyps:
        st.dataframe(hyps, width="stretch", hide_index=True)
    else:
        st.info("No verified hypothesis records yet — run the analyst (one record per "
                "paper survives only if its citation + every quote check out).")

# Security Policy

## Scope

This is a research/engineering project — an autonomous multi-agent system for
VLM optimization. It is **not** a production service and ships no network-facing
server. The most relevant security considerations are:

- **Local execution.** The runners, construction loop, and operator console run
  locally; the Streamlit console binds to `localhost` by default. Do not expose
  it to an untrusted network.
- **Backend credentials.** The agent backend is **local by default**
  (llama.cpp). The optional API backend reads an API key from the
  `ANTHROPIC_API_KEY` environment variable — it is never read from, written to,
  or committed in the repository. Keep keys in your environment or a secret
  manager, not in `run.yaml` or source.
- **Untrusted models/data.** Teacher models, datasets, and benchmarks are
  downloaded from third parties (see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)).
  Treat downloaded weights and data as untrusted input.

## Reporting a vulnerability

If you find a security issue, please **do not open a public issue**. Instead,
report it privately via GitHub's
[private vulnerability reporting](https://github.com/hanwook-cho/multi-agent-vlm-optimization-system/security/advisories/new)
("Report a vulnerability" under the **Security** tab).

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal repro or proof-of-concept if possible),
- the affected file(s) / commit.

This is a solo-maintained project, so response times are best-effort. You'll get
an acknowledgement as soon as the report is reviewed; please allow reasonable
time for a fix before any public disclosure.

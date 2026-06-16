# Local changes to the vendored VLMEvalKit

This directory is a vendored copy of [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
(Apache-2.0; see [`LICENSE`](LICENSE)). Per Apache-2.0 §4(b), this file records the
modifications made to the upstream sources in this repository:

- **decord patches** — made the `decord` video-decoding dependency optional / import-safe
  so the benchmark data-loading and scoring paths used by this project (image-only:
  POPE, MMBench, RealWorldQA) run on Apple-Silicon/macOS without the video stack.
  (Committed in `7ad3c35`, "vendor VLMEvalKit with decord patches".)

No other functional changes. The package is used only for benchmark **data loading and
scoring**; benchmark datasets themselves are downloaded by the user at run time from
their original hosts and are not redistributed here.

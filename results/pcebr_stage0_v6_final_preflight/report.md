# Stage-0 v6 final preflight

Date: 2026-10-16

Source attachment copied unchanged to stage0_pcebr_v6_final_original.py.

- Raw attachment size: 145448 bytes
- SHA256 after CRLF→LF normalization: 6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515
- Proposer-declared normalized SHA256: 6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515
- Hash status: matched.

The two prior numerical/control clarifications are implemented: FP32 dp @ V and active-kernel zero-noop with six-layer invocation assertion. The piecewise analytic positive/negative area formula is present.

Preflight remains open because the header claims a fine-grid quadrature validation absent from M0, and “100% exact replay” may mean output-level parity although the source does not compare against archived v5 logits/probabilities. Questions saved in questions_to_yoshua_bengio.md.

No source import, M0, model inference, training, or remote GPU job was run.

Later decision (2026-10-16): the user requested execution after discussing whether these two questions require proposer input. They are validation/reporting limitations that can be handled locally. The original source is preserved; a separate fine-grid/analytic audit and same-runtime original-v5 replay are included in the execution wrapper. See ../pcebr_stage0_v6/protocol.md. The status above records the historical preflight only.

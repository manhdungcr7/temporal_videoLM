# Presentation-time input correctness control — declared before execution

2026-10-15. Exactly the same frozen40, <=45s video and decoded audio. No replacement,
gold-driven changes, question parsing for routing or new spatial/temporal budget.
Use all110 cyclic answer rotations and identical native letter-only prompt.

P0 copies the original native CCLD C0 scores. P1 decodes with PyAV and samples
uniform rounded frame indices, using exactly the original utility's frame count,
output geometry and FPS. P2 uses the same PyAV decoder and selects the frame with
latest PTS <= each grid time i/native_fps. Common origin is container.start_time;
PTS are multiplied by their frame.time_base. Nonmonotone decoded PTS are stable
sorted, duplicate times choose last decoded picture. Before first frame, black;
after last, hold last picture. Record both cases. Original audio waveform is
passed through unchanged. No invented audio clock correction.

P1 vs P0 changes decoder and available decoded frame count as well as pixels;
it is not a pure decoder-only ablation. P2 vs P1 isolates selection rule under
the same decoded frames. P2 target times/processor FPS are identical to native,
so every returned frame has the time that processor assigns to its slot.

This is not an oracle: PyAV/FFmpeg are not independent event-time annotations.
Pixel audit suggests this control is needed, not that it must improve QA.
It is preprocessing correctness, not a novel CVPR-worthy reasoning method.

Only first rotation runs vision/audio encoders. Other choices share exactly
identical media embeddings (causal prefix), fresh lexical embeddings and native
position_ids. First decoder-vs-full-native replay must be <=.12, unchanged from
CCLD. No prefix KV approximation or hidden-state intervention in P1/P2.
Source PTS and selected indices are saved per clip. Main comparisons use all40.

References: https://pyav.org/docs/stable/api/time.html and Qwen's official
qwen_omni_utils/v2_5/vision_process.py. Local installed utility governs the
matched baseline (not whatever the GitHub main branch happens to contain).

## Transport recovery (recorded after server1 tunnel went offline)

At19/20 server1 aggregate records downloaded, endpoint returned ngrok offline
ERR_NGROK_3200. Server2 completed20/20. Recover EXACT missing ID1898 on server2,
same unchanged math source, original fixed sample and archived C0 reference.
Wrapper changes only sample assignment and reference-file paths. Save wrapper,
job/status and SHA in recovery_* files. No sample replacement or QA-based tuning.

All19 completed server1 aggregate scores/clock metadata/audit entries remain
local. Their per-rotation raw files were not downloaded before tunnel loss;
report this raw-archive gap explicitly instead of inventing individual scores.
This does not affect fully archived110 rotations of Argon CCLD or C/S/Q/L/E.

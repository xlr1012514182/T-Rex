# Synthetic RGB label audit

- Inspected on: 2026-08-17
- Episode: `bottle_000`
- Frames inspected: `frame_000000.png` and `frame_000019.png`
- Observed resolution: 160×120 RGB
- Observed convention: a centered blue rectangular proxy grows from a small pre-contact target to a larger near-contact target.
- Code locator: `revo3_v1/data/synthetic.py::_draw_frame`
- Metadata locator: generated `episodes/bottle_000/meta.json`

`labels_match=true` in the readiness manifest means only that the generated RGB follows this deterministic synthetic task/geometry convention. It does **not** mean that a real bottle is visible, that semantic object recognition works, or that the physical task succeeds.

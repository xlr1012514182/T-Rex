# Remote GPU bounded smoke summary

Outcome: **runnable-smoke-pass**. This is model-loading and one-forward evidence, not a paper, robot-task, or clinical result.

- Qwen exact revision loaded on one RTX 4080 SUPER. Native generation succeeded; the 128-token bounded run was deliberately truncated and therefore failed strict JSON. The 384-token strict run succeeded with a schema-valid `ASK_CLARIFY` response. Peak GPU use: 4540 MiB.
- Official T-Rex midtrain exact revision loaded with 1412 source keys, 0 missing and 0 unexpected keys. The production `CascadedServer` ran one full tactile `slow_and_fast` call and returned finite `[16, 62]` actions. Server call: 1220.31 ms; GPU peak: 8674 MiB; host-memory peak delta: 25233 MiB.
- The T-Rex result is the official dual-hand 62D embodiment. No Revo3 21D compatibility claim is made.
- Source deviations are limited to a bounded exit before ZMQ bind and a task-only observability wrapper.

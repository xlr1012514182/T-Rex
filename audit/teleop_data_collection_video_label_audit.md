# Synthetic frame/label audit

The four generated fixture episodes were inspected at the schema/provenance level:

- each episode contains four readable `64x48` RGB frames;
- each frame slot maps to the same episode and 30 Hz causal anchor recorded in its metadata;
- the episode task key and instruction match the requested fixture (`bottle`, `phone`, `plastic_bag`, or `refrigerator_door`);
- every exported hand action equals the single-use Revo controller `exact_sent_target` attached to that anchor.

This check establishes serialization and slot-label consistency only. The images are synthetic fixtures, so it is not evidence that a real object is visible, that the motion is expert-quality, or that any task succeeds.

# Contributing

Thank you for contributing to T-Rex × Revo 3. Keep changes scoped, documented and testable.

## Development

Use Python 3.10 and install `requirements-dev.txt` from the repository root. See the [development guide](docs/DEVELOPMENT.md) for optional training and hardware environments.

```bash
python -m pytest -q
python scripts/check_release.py
```

## Change checklist

- Add regression coverage for changed interfaces and behavior.
- Preserve task/version binding, causal timestamps, exact-sent action labels and strict artifact identity checks.
- Keep hardware writes opt-in. Do not bypass arming, freshness checks, calibrated limits, stop confirmation or physical safety procedures.
- Label simulated devices and synthetic datasets explicitly; report measurements with their actual setup and scope.
- Use relative paths or configurable environment variables in examples. Do not commit local device identities, calibration files, model checkpoints, datasets, credentials or run outputs.
- Clear notebook outputs before submission.
- Update the relevant guide when changing CLI flags, schemas or configuration.
- Preserve upstream attribution and component-specific licenses. Public availability does not by itself grant redistribution rights.

## Proposing a change

Describe the problem, the changed behavior and the commands used to check it. For a hardware-facing change, include the interface assumptions and unit conversions without sharing local deployment details. Keep reproducible unit tests separate from device-specific acceptance records.

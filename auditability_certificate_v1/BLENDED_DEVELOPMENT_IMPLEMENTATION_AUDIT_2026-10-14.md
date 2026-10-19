# Blended development implementation audit — 2026-10-14

## Disposition

The frozen seed-4400 run is valid as a victim-only development experiment under
its own declared implementation. Hashes of the configuration, wrapper, shared
training engine, architecture, trigger asset, BackdoorBench reference config,
and reference transform match the freeze manifest. Poison indices, assigned
labels, training transforms, and non-target ASR evaluation are aligned. The
runner does not import or compute detector or SRAC endpoints.

## Naming boundary

The attack must be called **BackdoorBench-parameterized Blended** or a
**Blended variant**, not an exact execution of every BackdoorBench default. The
frozen protocol intentionally uses:

- the published default train/test blend strength of 0.2;
- a hash-checked BackdoorBench Hello-Kitty trigger asset;
- target class 8 and a 5% training-row poison rate chosen for this experiment;
- direct floating-point blending after augmentation.

BackdoorBench's referenced default YAML instead names target class 0 and poison
rate 0.1. Its referenced transform also performs an image/uint8 round trip and
names a JPEG asset, whereas this frozen implementation uses the packaged PNG
asset and no uint8 truncation. These are disclosed implementation choices, not
evidence that seed 4400 is invalid.

If the development gate passes, formal confirmation must either preserve this
exact frozen attack transform and use the bounded name above, or start a new
development protocol before claiming exact upstream BackdoorBench fidelity.


# orbital-py (development version)

## Internal cleanups

- **Optimizer:** resolved a long-standing TODO around constant-folding of
  `Gather` / `GatherND` outputs in `src/orbital/translation/optimizer.py`;
  the optimizer now recognises the folded shapes without a bespoke
  side-channel. (#P-01)
- **Translator:** removed the FIXME scaffolding in
  `src/orbital/translation/translator.py` around initializer deduplication;
  the deduplication path is now the default and documented. (#P-02)
- **ONNX utilities:** tightened type handling in
  `src/orbital/_utils/onnx.py` (formerly marked TODO) so tensor-element
  types are resolved through a single lookup table. (#P-03)
- **Gather step:** addressed the TODO in
  `src/orbital/translation/steps/gather.py` covering negative-index
  normalisation; behaviour is unchanged but the logic is no longer
  conditional on a runtime check. (#P-04)
- **LinearRegressor step:** cleaned up the FIXME in
  `src/orbital/translation/steps/linearreg.py` so the multi-target path
  shares the single-target coefficient expansion helper. (#P-05)

No user-visible API changes in this release.

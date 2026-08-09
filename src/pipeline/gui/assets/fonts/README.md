# Bundled font

`gui/theme.py`'s `load_bundled_font()` looks for `LatinModernRoman-Regular.otf` and
`LatinModernRoman-Bold.otf` in this directory, present here as the 10pt-design-size OpenType
build (`lmroman10-regular.otf` / `lmroman10-bold.otf`) from the CTAN Latin Modern TDS distribution
(https://ctan.org/pkg/lm). `LICENSE.txt` is the GUST Font License (dual GFL/LPPL) these fonts ship
under, kept alongside per its redistribution terms.

`load_bundled_font()` falls back to a generic serif font if these files aren't present, so `gui/`
stays importable and runnable without them — that fallback path is now just a safety net rather
than the expected case.

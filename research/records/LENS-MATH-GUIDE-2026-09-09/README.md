# From reaction rates to model lenses

A phone-readable PDF written for a chemistry/biochemistry reader who understands derivatives and wants the linear-algebra vocabulary grounded in concrete examples.

**Deliverable:** `output/pdf/lens-mathematics-a-grounded-guide.pdf` (18 small portrait pages). The PDF has large body text, typeset equations, chemical worked examples, three illustrative plots, a glossary, PDF bookmarks and clickable academic/source references. The downloadable copy lives in the existing private repository; it is not publicly published.

The examples and narrative separate mathematical illustration, recorded model measurements and unexecuted controls. Source research is frozen at the numerical diagnosis `558d424` and saturation audit `6983d8e`; this document performs no model work.

## Rebuild

1. Run `make_figures.py` with Matplotlib, NumPy and Pillow. It produces equation/plot PNGs in `tmp/pdfs/lens-math-guide/assets`.
2. Run `build_pdf.py` with ReportLab and Pillow. The manuscript is `content.py`. The default build refuses any page whose content intrudes into the footer area.
3. Run `verify_pdf.py` with pypdf, after the build. Render the PDF with Poppler and inspect all pages.

The local build used the bundled Codex Python runtime for ReportLab/pypdf, and the existing isolated plotting runtime for Matplotlib. `build_pdf.py` points to the existing DejaVu font directory; change that font directory when reproducing on another computer. PDF page size is 420 × 780 points, with 13.2-point body type.

## Verification

All worked examples were independently checked. `VERIFICATION.json` records the PDF hash and checks of its page count, bookmarks, titles, footers, links, layout boundaries and exact-arithmetic teaching examples. All pages were visually inspected; a duplicate diagram occupying an equation slot was caught and corrected before delivery. No source experiment, checkpoint, model process or lock was changed.

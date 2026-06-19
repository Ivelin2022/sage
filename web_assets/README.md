# SAGE web assets

Drop-in SVGs for the website (and the paper/slides). All use the brand palette:
teal `#26c6da` (primary), cyan `#4fc3f7` (interactive), coral `#ff7043` (the one
"we report the negatives" accent), on the `#0a0a0a` dark base.

| File | Use |
|---|---|
| `sage_logo.svg` | Wordmark + memory-lattice mark. Hero and footer. Transparent bg (works on dark). |
| `results_chart_dark.svg` | "Honest results" chart for the **dark site**. SAGE-pc below the per-class-k-means line; SAGE-grid collapses. |
| `results_chart_light.svg` | Same chart, **white background**, for the paper / slides. |
| `poster1_assistant.svg` | Case-study 1 card image / **video poster** (800×460, terminal style with a play affordance). |
| `poster2_drone.svg` | Case-study 2 card image / video poster. |
| `poster3_continual.svg` | Case-study 3 card image / video poster. |

## Notes
- **Posters are placeholders.** They make the three case-study cards look complete
  before the GIFs/MP4s are recorded. Once you record the demos (ScreenToGif or OBS),
  swap each poster for the clip — keep the poster as the video's `poster=` frame.
- **Need PNGs instead of SVG?** Open the SVG in a browser and screenshot, or use any
  SVG→PNG export (e.g. Inkscape, or an online converter). Export at 2× for retina.
- **Recolour:** every colour is a literal hex in the file — search/replace to retune.
- The case-study copy text lives in `../website_brief.md`; the page structure brief
  is there too.

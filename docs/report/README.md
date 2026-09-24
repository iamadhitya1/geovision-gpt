# Report build

`build_report.js` generates the B.Tech project report (`BTP_Report_VQA_RemoteSensing.docx`)
with the `docx` npm package. To regenerate after editing the text:

```bash
cd docs/report
npm install
node build_report.js     # writes BTP_Report_VQA_RemoteSensing.docx here
```

The `.docx` itself is not committed (it's a build output); copy it wherever it's needed.
Keep the numbers in sync with `ml/README.md` and `frontend/src/lib/results.ts`.

## Figures

`figures/` holds the source images the report embeds (architecture diagram, results
chart, and real screenshots of the running app) — these ARE committed, since they're
inputs, not build output. To regenerate a screenshot after a UI change, drive the
running app with Puppeteer (`puppeteer-core`, pointed at a local Chrome install) and
crop with `sharp`; see this session's history for the working script pattern, or
just take fresh screenshots and drop them in `figures/` under the same filenames.

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

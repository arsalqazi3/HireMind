# HireMind demo

From the repository root, start the Streamlit app with the base checkpoint and the locally prepared catalogue, index, and model cache.

```bash
HIREMIND_ADAPTER_DIR= .venv/bin/python -m streamlit run app/streamlit_app.py
```

1. Upload the fictional data analyst résumé at `data/samples/sample_data_analyst.txt` and choose **Find matching jobs**. Leave **In-depth review** off for the main demo.
2. Open one result and show the score breakdown and posting evidence. Open **Search steps** to show the tool calls and any retry.
3. Select a returned posting in the cover-letter panel. Generate a draft, check the résumé evidence it uses, and show that the draft is editable. Keep the local Ollama server running with `qwen3:8b` available to demonstrate the AI-written body.

Use `outputs/demo/HireMind-demo.webm` as the recorded backup. The first live run includes model loading, so rehearse once before presenting. The catalogue uses historical postings and the demo can return fewer than five when candidates fall below the qualification cutoff.

The November 27 smoke run used the current base workflow and the fictional data analyst résumé. It returned five unique jobs, recorded two search steps, and produced a cover-letter draft. Matching took 167 seconds on the Mac GPU with the local AI review running concurrently. The timing covers one run under that load. The saved result is `outputs/demo/current-base-smoke.json`.

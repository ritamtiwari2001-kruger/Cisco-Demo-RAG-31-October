# Cisco Nexus 9000 Troubleshooting Assistant

A small, local Retrieval-Augmented Generation (RAG) demo. It indexes the Cisco
Nexus 9000 troubleshooting PDF in ChromaDB, retrieves the five most relevant
passages, and uses Groq GPT-OSS 120B to answer questions in a Gradio chat UI.

## Requirements

- Windows, Python 3.12, and an internet connection for package/model downloads
  and Groq inference.
- A Groq API key in the project `.env` file.
- The source PDF at
  `data/cisco_nexus_9000_troubleshooting_guide.pdf`.

The app's first run downloads the local `all-MiniLM-L6-v2` embedding model and
creates `chroma_db/`. Later runs reuse the index unless the PDF has changed.

## Setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

The interpreter path must be inside this project's `.venv` directory. Add
`GROQ_API_KEY=your_key` to `.env`; do not commit that file. `.env.example`
shows the expected variable name.

Put the PDF in `data\cisco_nexus_9000_troubleshooting_guide.pdf`, then start:

```powershell
python app.py
```

Open http://127.0.0.1:7860. Conversation history is kept for the current
browser session only.

## Tests

Run the local unit tests without making Groq API calls:

```powershell
python -m unittest discover -s tests -v
```

For the factual and conceptual end-to-end smoke tests, start the app and submit:

1. `What does PORT-3-IF_UNSUPPORTED_TRANSCEIVER indicate, and what
   troubleshooting action does the guide recommend?` The guide describes this
   message and its recommended action in the supplied PDF. Confirm the answer
   matches the passage and cites its page.
2. `Based on the Cisco Nexus 9000 troubleshooting guide, describe a sensible
   step-by-step approach an engineer can follow to investigate a recurring
   network or switch issue. Explain how the engineer can use the documented
   symptoms and troubleshooting procedures to narrow down the possible cause.`

Both use the same top-five retrieval flow. End-to-end answers require a working
Groq API key and network access.

## Rebuild the index

Replace the PDF at the expected path and restart the app. A SHA-256 source
fingerprint triggers re-indexing automatically. To force a rebuild, stop the
app and remove the project's `chroma_db` directory before restarting.

## Troubleshooting

- **Missing PDF:** Check the exact filename and `data` folder path above.
- **Missing API key:** Add `GROQ_API_KEY` to `.env`.
- **Model/network errors:** Check internet access and the Groq model/API
  availability; initialization details are logged in the terminal.
- **No extracted text:** This POC does not OCR scanned/image-only PDFs.
- **PowerShell activation blocked:** Use
  `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and
  `\.venv\Scripts\python.exe app.py` without changing the execution policy.

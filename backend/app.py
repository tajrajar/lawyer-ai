"""
FastAPI web server for the Pakistani Legal AI analyzer.

Serves a browser upload UI and a POST /analyze endpoint that extracts PDF text,
retrieves relevant law context via RAG, and returns structured LLM analysis.

Usage: uvicorn app:app --reload --port 8000
"""
import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
load_dotenv()

from data.pdf_engine  import extract_text_auto
from data.searcher    import get_searcher
from core.analyzer    import analyze_async
from core.router      import check_providers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pakistani Legal AI",
    description="AI-powered Pakistani law document analyzer",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Professional Browser UI ──────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page browser UI for PDF upload and analysis."""
    return """
    <html>
    <head>
        <title>Pakistani Legal AI</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px;
                   margin: 40px auto; padding: 20px; background: #eceff1; color: #333; }
            h1 { color: #085041; text-align: center; border-bottom: 3px solid #085041; padding-bottom: 10px; }
            .box { background: white; padding: 25px; border-radius: 12px;
                   margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 15px; }
            label { font-weight: bold; display: block; margin-bottom: 5px; }
            select, button, input[type=file] { width: 100%; padding: 12px; margin-top: 5px; border-radius: 6px; border: 1px solid #ccc; }
            button { background: #085041; color: white; border: none; font-weight: bold; cursor: pointer; font-size: 16px; transition: 0.3s; }
            button:hover { background: #0b7a63; }
            #result { white-space: pre-wrap; font-size: 14px; line-height: 1.6; background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 5px solid #085041; }
            .loading { color: #d35400; font-weight: bold; text-align: center; display: block; }
        </style>
    </head>
    <body>
        <h1>⚖️ Pakistani Legal AI Analyzer</h1>
        <p style="text-align:center;">PPC | CrPC | Constitution | Local OpenClaw Support</p>

        <div class="box">
            <div class="form-group">
                <label>1. Case PDF Upload Karein:</label>
                <input type="file" id="pdf" accept=".pdf">
            </div>

            <div class="form-group">
                <label>2. Zuban (Language) Select Karein:</label>
                <select id="language">
                    <option value="Urdu">Urdu (اردو)</option>
                    <option value="English">English</option>
                </select>
            </div>

            <div class="form-group">
                <label>3. AI Engine Select Karein:</label>
                <select id="provider">
                    <option value="claude">Claude 3.5 (Best Accuracy)</option>
                    <option value="groq">Groq (Fastest)</option>
                    <option value="openclaw">OpenClaw (Private/Local)</option>
                    <option value="openai">OpenAI GPT-4o</option>
                    <option value="gemini">Google Gemini</option>
                </select>
            </div>

            <button onclick="analyze()">Analyze Legal Document</button>
        </div>

        <div class="box">
            <h3>Analysis Result:</h3>
            <div id="result">Result yahan nazar ayega...</div>
        </div>

        <script>
        async function analyze() {
            const file = document.getElementById('pdf').files[0];
            if (!file) { alert('Meharbani karke PDF file select karein'); return; }

            const resDiv = document.getElementById('result');
            resDiv.innerHTML = '<span class="loading">⏳ Qanooni tajzia (Analysis) jari hai... Meharbani karke intezar karein.</span>';

            const form = new FormData();
            form.append('file', file);
            form.append('target_lang', document.getElementById('language').value);
            form.append('provider', document.getElementById('provider').value);

            try {
                const response = await fetch('/analyze', {method:'POST', body:form});
                const data = await response.json();

                if (!response.ok) {
                    resDiv.textContent = 'Error: ' + (data.detail || 'Server error');
                    return;
                }

                // Formatting Result Output
                const asList = (items, label) => {
                    if (!Array.isArray(items)) {
                        return `  (Couldn't parse ${label})\\n`;
                    }
                    if (items.length === 0) return `  (None)\\n`;
                    return items.map(p => `  • ${p}\\n`).join('');
                };

                let out = `📌 CASE TYPE: ${data.case_type || 'N/A'}\\n`;
                out += `🌐 LANGUAGE: ${data.target_lang || 'N/A'}\\n`;
                out += `📊 CASE STRENGTH: ${data.case_strength ?? 'N/A'}%\\n`;
                out += `--------------------------------------------------\\n`;
                out += `📝 SUMMARY:\\n${data.summary || '(No summary)'}\\n\\n`;

                out += `✅ STRONG POINTS:\\n`;
                out += asList(data.strong_points, 'strong points');

                out += `\\n❌ WEAK POINTS:\\n`;
                out += asList(data.weak_points, 'weak points');

                out += `\\n⚖️ RELEVANT SECTIONS (Verified):\\n`;
                if (!Array.isArray(data.relevant_sections)) {
                    out += `  (Couldn't parse relevant sections)\\n`;
                } else if (data.relevant_sections.length === 0) {
                    out += `  (None)\\n`;
                } else {
                    data.relevant_sections.forEach(s => {
                        out += `  [${s.ref || '?'}] ${s.title || ''}\\n    Reason: ${s.reason || ''}\\n`;
                    });
                }

                resDiv.textContent = out;
            } catch(e) {
                resDiv.textContent = 'Technical Error: ' + e.message;
            }
        }
        </script>
    </body>
    </html>
    """

# ── Analyze Endpoint ──────────────────────────────────────────
@app.post("/analyze")
async def analyze_pdf(
    file:        UploadFile = File(...),
    target_lang: str        = Form("Urdu"),
    provider:    str        = Form("claude"),
    mask_pii:    bool       = Form(True),
):
    """
    Accept a case PDF, extract text, retrieve law context, and run LLM analysis.

    Args:
        file: Uploaded PDF document.
        target_lang: Output language for narrative fields ("Urdu" or "English").
        provider: LLM provider key (groq, openai, gemini, claude, openclaw).
        mask_pii: Whether to mask CNIC, phone, email, and account numbers.

    Returns:
        Structured analysis JSON from core.analyzer.analyze_async.

    Raises:
        HTTPException 400: Non-PDF file uploaded.
        HTTPException 422: Text extraction yielded insufficient content.
        HTTPException 500: Unexpected processing error.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf PDF format allow hai.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(await file.read())
            tmp_path = f.name

        # 1. Text Extraction (auto-retries with OCR for scanned PDFs)
        text = extract_text_auto(tmp_path)
        if len(text.strip()) < 50:
            raise HTTPException(422, "PDF se text nahi nikal saka. Kya ye scanned image hai?")

        # 2. RAG Context (Verified Laws)
        law_context = ""
        searcher = get_searcher()
        if searcher:
            results = searcher.search(text[:2000], top_k=5)
            law_context = "\n".join([f"Law: {r['law']} - {r['text']}" for r in results])

        # 3. AI Analysis Call
        result = await analyze_async(
            doc_text=text,
            law_context=law_context,
            provider=provider,
            mask_pii=mask_pii,
            target_lang=target_lang,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backend Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
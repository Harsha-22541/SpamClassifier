"""Interactive Flask Web Application for Naive Bayes Spam/Ham Classifier with Word Influence Reporting."""

from flask import Flask, request, jsonify, render_template_string
import os
from src.naive_bayes import NaiveBayesClassifier
from src.preprocessor import TextPreprocessor
from src.dataset import CURATED_BENCHMARK_EMAILS, load_sms_dataset, create_curated_email_dataset

app = Flask(__name__)

# Load or train model
MODEL_PATH = "models/naive_bayes_model.json"
preprocessor = TextPreprocessor(lowercase=True, normalize_urls=True, normalize_currency=True)

if os.path.exists(MODEL_PATH):
    model = NaiveBayesClassifier.load(MODEL_PATH, preprocessor=preprocessor)
else:
    sms_texts, sms_labels = load_sms_dataset("data/sms_dataset.tsv")
    email_texts, email_labels = create_curated_email_dataset("data/emails.csv")
    model = NaiveBayesClassifier(alpha=1.0, preprocessor=preprocessor)
    model.fit(sms_texts + email_texts, sms_labels + email_labels)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Naive Bayes Email Spam Classifier | Word Influence Explainer</title>
    <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .token-spam {
            background-color: rgba(239, 68, 68, 0.25);
            color: #fca5a5;
            border-bottom: 2px solid #ef4444;
            padding: 2px 4px;
            border-radius: 3px;
            cursor: pointer;
            position: relative;
        }
        .token-ham {
            background-color: rgba(34, 197, 94, 0.25);
            color: #86efac;
            border-bottom: 2px solid #22c55e;
            padding: 2px 4px;
            border-radius: 3px;
            cursor: pointer;
            position: relative;
        }
        .token-neutral {
            color: var(--text-muted);
            padding: 2px 2px;
        }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <div class="max-w-6xl mx-auto space-y-6">

        <!-- Top Header -->
        <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-[var(--border-color)] gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="px-2.5 py-1 rounded bg-indigo-500/20 text-indigo-400 font-mono text-xs uppercase font-semibold tracking-wider">Explainable AI</span>
                    <span class="text-xs text-slate-400 font-mono">Multinomial Naive Bayes</span>
                </div>
                <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight mt-1 text-white">
                    Email Spam / Ham Classifier
                </h1>
                <p class="text-sm text-slate-400 mt-1">
                    Exact Log-Odds Word Influence Attribution & Heatmap Decomposition
                </p>
            </div>
            <div class="flex items-center gap-3">
                <div class="text-right">
                    <div class="text-xs text-slate-400">Model Vocabulary</div>
                    <div class="text-sm font-bold text-white font-mono">{{ vocab_size }} tokens</div>
                </div>
                <div class="h-8 w-[1px] bg-slate-700"></div>
                <div class="text-right">
                    <div class="text-xs text-slate-400">Baseline Prior</div>
                    <div class="text-sm font-bold text-indigo-400 font-mono">{{ "%.3f"|format(prior_log_odds) }}</div>
                </div>
            </div>
        </header>

        <!-- Presets Row -->
        <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-4">
            <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider mb-2.5 flex items-center gap-2">
                <span>Select a Preset Email Scenario:</span>
            </div>
            <div class="flex flex-wrap gap-2" id="preset-buttons">
                <!-- Injected via JS -->
            </div>
        </div>

        <!-- Main Grid: Input Form & Results -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left: Email Input (5 Cols) -->
            <div class="lg:col-span-5 space-y-4">
                <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-5 shadow-sm space-y-4">
                    <h2 class="text-base font-semibold text-white flex items-center justify-between">
                        <span>Email Content</span>
                        <span class="text-xs text-slate-400 font-normal">Subject & Body</span>
                    </h2>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Subject</label>
                        <input type="text" id="email-subject" placeholder="e.g. URGENT: Verify account"
                            class="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors">
                    </div>

                    <div>
                        <label class="block text-xs font-medium text-slate-300 mb-1">Email Body</label>
                        <textarea id="email-body" rows="9" placeholder="Paste or type the full email message text..."
                            class="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-indigo-500 font-sans transition-colors resize-y"></textarea>
                    </div>

                    <div class="flex items-center justify-between pt-2">
                        <button onclick="clearInput()" class="px-4 py-2 text-xs font-medium text-slate-400 hover:text-white transition-colors">
                            Clear
                        </button>
                        <button onclick="classifyEmail()" id="btn-classify"
                            class="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-sm rounded-lg shadow transition-all flex items-center gap-2">
                            <span>Classify & Explain</span>
                            <span id="spinner" class="hidden animate-spin">&#9696;</span>
                        </button>
                    </div>
                </div>

                <!-- Mathematical Formula Explainer Card -->
                <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-5 text-xs text-slate-400 space-y-2">
                    <div class="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">How Decision Influence is Computed:</div>
                    <div class="font-mono bg-slate-900 p-2.5 rounded border border-slate-800 text-slate-300 text-[11px] leading-relaxed">
                        Log-Odds(&Delta;) = Prior + &sum; [ Count(w) &times; LLR(w) ]<br>
                        LLR(w) = log P(w | Spam) - log P(w | Ham)
                    </div>
                    <ul class="list-disc list-inside space-y-1 text-slate-400 pt-1">
                        <li><strong class="text-red-400">LLR > 0</strong>: Word increases Spam probability</li>
                        <li><strong class="text-emerald-400">LLR < 0</strong>: Word increases Ham probability</li>
                        <li>Every word's contribution is <strong class="text-white">exact and additive</strong></li>
                    </ul>
                </div>
            </div>

            <!-- Right: Results & Word Influence (7 Cols) -->
            <div class="lg:col-span-7 space-y-6">

                <!-- Decision Banner Card -->
                <div id="results-card" class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-6 shadow-sm space-y-5">
                    
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-700/60 gap-4">
                        <div>
                            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Classification Decision</div>
                            <div class="flex items-center gap-3 mt-1">
                                <span id="badge-prediction" class="text-2xl font-black px-4 py-1 rounded-lg uppercase tracking-wide">
                                    --
                                </span>
                                <span id="badge-confidence" class="text-sm font-semibold text-slate-300">
                                    Confidence: --%
                                </span>
                            </div>
                        </div>

                        <!-- Mini gauge stats -->
                        <div class="grid grid-cols-2 gap-3 text-center sm:text-right">
                            <div class="bg-slate-900 px-3 py-2 rounded-lg border border-slate-800">
                                <div class="text-[10px] uppercase font-bold text-red-400">Spam Prob</div>
                                <div id="stat-spam-prob" class="text-base font-bold text-red-400 font-mono">--%</div>
                            </div>
                            <div class="bg-slate-900 px-3 py-2 rounded-lg border border-slate-800">
                                <div class="text-[10px] uppercase font-bold text-emerald-400">Ham Prob</div>
                                <div id="stat-ham-prob" class="text-base font-bold text-emerald-400 font-mono">--%</div>
                            </div>
                        </div>
                    </div>

                    <!-- Log-Odds Bar Breakdown -->
                    <div>
                        <div class="flex justify-between text-xs text-slate-400 mb-1 font-mono">
                            <span>Ham Evidence (&Delta; &le; 0)</span>
                            <span id="stat-log-odds" class="text-white font-bold">&Delta; = 0.000</span>
                            <span>Spam Evidence (&Delta; > 0)</span>
                        </div>
                        <div class="w-full bg-slate-900 h-3 rounded-full overflow-hidden flex border border-slate-800">
                            <div id="bar-ham" class="bg-emerald-500 h-full transition-all duration-500" style="width: 50%"></div>
                            <div id="bar-spam" class="bg-red-500 h-full transition-all duration-500" style="width: 50%"></div>
                        </div>
                    </div>

                    <!-- Highlighted Token Heatmap -->
                    <div>
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="text-xs font-semibold uppercase text-slate-400 tracking-wider">
                                Email Word Influence Heatmap
                            </h3>
                            <span class="text-[11px] text-slate-400">Click or hover on highlighted words</span>
                        </div>
                        <div id="email-heatmap"
                            class="p-4 bg-slate-900 rounded-lg border border-slate-800 font-mono text-sm leading-relaxed max-h-52 overflow-y-auto whitespace-pre-wrap select-text">
                            Run classification to view word influence heatmap...
                        </div>
                    </div>

                    <!-- Top Influencing Words Table -->
                    <div>
                        <h3 id="table-influencing-title" class="text-xs font-semibold uppercase text-slate-400 tracking-wider mb-2.5">
                            Top Words Driving This Classification
                        </h3>
                        <div class="overflow-x-auto border border-slate-800 rounded-lg bg-slate-900/60">
                            <table class="w-full text-left text-xs">
                                <thead>
                                    <tr class="border-b border-slate-800 bg-slate-900 text-slate-400 uppercase font-mono">
                                        <th class="py-2 px-3">#</th>
                                        <th class="py-2 px-3">Word</th>
                                        <th class="py-2 px-3 text-center">Freq</th>
                                        <th class="py-2 px-3 text-right">LLR (Weight)</th>
                                        <th class="py-2 px-3 text-right">Net Impact</th>
                                        <th class="py-2 px-3">Share</th>
                                    </tr>
                                </thead>
                                <tbody id="table-influential-body" class="divide-y divide-slate-800/60 font-mono">
                                    <tr>
                                        <td colspan="6" class="py-4 text-center text-slate-500 font-sans">
                                            No classification executed yet.
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Counter Evidence Words -->
                    <div id="counter-section">
                        <h3 id="table-counter-title" class="text-xs font-semibold uppercase text-slate-400 tracking-wider mb-2.5">
                            Opposing Evidence (Mitigating Words)
                        </h3>
                        <div class="overflow-x-auto border border-slate-800 rounded-lg bg-slate-900/60">
                            <table class="w-full text-left text-xs">
                                <thead>
                                    <tr class="border-b border-slate-800 bg-slate-900 text-slate-400 uppercase font-mono">
                                        <th class="py-2 px-3">#</th>
                                        <th class="py-2 px-3">Word</th>
                                        <th class="py-2 px-3 text-center">Freq</th>
                                        <th class="py-2 px-3 text-right">LLR (Weight)</th>
                                        <th class="py-2 px-3 text-right">Opposing Impact</th>
                                        <th class="py-2 px-3">Share</th>
                                    </tr>
                                </thead>
                                <tbody id="table-counter-body" class="divide-y divide-slate-800/60 font-mono">
                                    <tr>
                                        <td colspan="6" class="py-3 text-center text-slate-500 font-sans">
                                            None
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                </div>
            </div>

        </div>
    </div>

    <!-- Client-side Logic -->
    <script>
        const PRESETS = {{ presets_json|safe }};

        function initPresets() {
            const container = document.getElementById('preset-buttons');
            PRESETS.forEach((p, idx) => {
                const btn = document.createElement('button');
                const isSpam = p.label === 'spam';
                btn.className = `px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-1.5 ${
                    isSpam 
                    ? 'border-red-500/30 text-red-300 hover:bg-red-500/10' 
                    : 'border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/10'
                }`;
                btn.innerHTML = `<span class="w-1.5 h-1.5 rounded-full ${isSpam ? 'bg-red-400' : 'bg-emerald-400'}"></span> ${p.subject.substring(0, 32)}...`;
                btn.onclick = () => loadPreset(p);
                container.appendChild(btn);
            });
        }

        function loadPreset(preset) {
            document.getElementById('email-subject').value = preset.subject;
            document.getElementById('email-body').value = preset.body;
            classifyEmail();
        }

        function clearInput() {
            document.getElementById('email-subject').value = '';
            document.getElementById('email-body').value = '';
        }

        async function classifyEmail() {
            const subject = document.getElementById('email-subject').value.trim();
            const body = document.getElementById('email-body').value.trim();
            const fullText = subject ? `Subject: ${subject}\\n\\n${body}` : body;

            if (!fullText) {
                alert("Please enter an email subject or body to classify.");
                return;
            }

            const spinner = document.getElementById('spinner');
            spinner.classList.remove('hidden');

            try {
                const res = await fetch('/api/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: fullText})
                });
                const data = await res.json();
                renderResults(data);
            } catch (err) {
                alert("Classification error: " + err);
            } finally {
                spinner.classList.add('hidden');
            }
        }

        function renderResults(data) {
            const isSpam = data.predicted_class === 'spam';
            const badge = document.getElementById('badge-prediction');
            badge.innerText = data.predicted_class.toUpperCase();
            if (isSpam) {
                badge.className = "text-2xl font-black px-4 py-1 rounded-lg uppercase tracking-wide bg-red-500/20 text-red-400 border border-red-500/30";
            } else {
                badge.className = "text-2xl font-black px-4 py-1 rounded-lg uppercase tracking-wide bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
            }

            document.getElementById('badge-confidence').innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            document.getElementById('stat-spam-prob').innerText = `${(data.probabilities.spam * 100).toFixed(1)}%`;
            document.getElementById('stat-ham-prob').innerText = `${(data.probabilities.ham * 100).toFixed(1)}%`;

            // Bar breakdown
            const spamPercent = Math.min(Math.max(data.probabilities.spam * 100, 2), 98);
            document.getElementById('bar-spam').style.width = `${spamPercent}%`;
            document.getElementById('bar-ham').style.width = `${100 - spamPercent}%`;
            document.getElementById('stat-log-odds').innerText = `Δ = ${data.total_log_odds > 0 ? '+' : ''}${data.total_log_odds.toFixed(3)}`;

            // Render Heatmap
            renderHeatmap(data);

            // Render Tables
            renderTables(data);
        }

        function renderHeatmap(data) {
            const container = document.getElementById('email-heatmap');
            const wordMap = {};
            data.all_words_breakdown.forEach(w => {
                wordMap[w.word] = w;
            });

            const rawText = data.text;
            // Split into tokens & punctuation
            const parts = rawText.split(/(\\b[a-zA-Z_][a-zA-Z0-9_]*\\b)/);

            const nodes = parts.map(segment => {
                const lower = segment.toLowerCase();
                if (wordMap[lower]) {
                    const w = wordMap[lower];
                    const contrib = w.contribution_to_spam;
                    const llr = w.log_likelihood_ratio;
                    const isSpamToken = contrib > 0.05;
                    const isHamToken = contrib < -0.05;

                    const span = document.createElement('span');
                    if (isSpamToken) {
                        span.className = 'token-spam';
                    } else if (isHamToken) {
                        span.className = 'token-ham';
                    } else {
                        span.className = 'token-neutral';
                    }
                    span.title = `Word: "${segment}" | LLR: ${llr.toFixed(3)} | Impact: ${contrib.toFixed(3)}`;
                    span.textContent = segment;
                    return span.outerHTML;
                } else {
                    return escapeHtml(segment);
                }
            });

            container.innerHTML = nodes.join('');
        }

        function renderTables(data) {
            const isSpam = data.predicted_class === 'spam';
            document.getElementById('table-influencing-title').innerText = 
                `Top Words Driving Decision Towards [${isSpam ? 'SPAM' : 'HAM'}]`;

            const tbodyInf = document.getElementById('table-influential-body');
            tbodyInf.innerHTML = '';

            data.top_influential_words.forEach((item, idx) => {
                const tr = document.createElement('tr');
                const impact = isSpam ? item.contribution_to_spam : -item.contribution_to_spam;
                const share = item.evidence_share || 0;
                const colorCls = isSpam ? 'text-red-400' : 'text-emerald-400';
                const barColor = isSpam ? 'bg-red-500' : 'bg-emerald-500';

                tr.innerHTML = `
                    <td class="py-2 px-3 text-slate-500">#${idx + 1}</td>
                    <td class="py-2 px-3 font-semibold text-white font-sans">${escapeHtml(item.word)}</td>
                    <td class="py-2 px-3 text-center text-slate-300">${item.count}</td>
                    <td class="py-2 px-3 text-right text-slate-400 font-mono">${(item.log_likelihood_ratio > 0 ? '+' : '')}${item.log_likelihood_ratio.toFixed(3)}</td>
                    <td class="py-2 px-3 text-right ${colorCls} font-bold font-mono">${(impact > 0 ? '+' : '')}${impact.toFixed(3)}</td>
                    <td class="py-2 px-3">
                        <div class="flex items-center gap-2">
                            <div class="w-16 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                <div class="${barColor} h-full" style="width: ${Math.min(share, 100)}%"></div>
                            </div>
                            <span class="text-[11px] text-slate-400">${share.toFixed(1)}%</span>
                        </div>
                    </td>
                `;
                tbodyInf.appendChild(tr);
            });

            // Counter evidence
            const oppName = isSpam ? 'HAM' : 'SPAM';
            document.getElementById('table-counter-title').innerText = 
                `Opposing Evidence (Words Pulling Towards [${oppName}])`;

            const tbodyCounter = document.getElementById('table-counter-body');
            tbodyCounter.innerHTML = '';

            if (data.top_opposing_words.length === 0) {
                tbodyCounter.innerHTML = `<tr><td colspan="6" class="py-2 px-3 text-slate-500 italic font-sans">None -- all tokens supported the predicted class or were neutral.</td></tr>`;
            } else {
                data.top_opposing_words.forEach((item, idx) => {
                    const tr = document.createElement('tr');
                    const oppImpact = isSpam ? -item.contribution_to_spam : item.contribution_to_spam;
                    const share = item.evidence_share || 0;
                    const colorCls = isSpam ? 'text-emerald-400' : 'text-red-400';

                    tr.innerHTML = `
                        <td class="py-2 px-3 text-slate-500">#${idx + 1}</td>
                        <td class="py-2 px-3 font-semibold text-white font-sans">${escapeHtml(item.word)}</td>
                        <td class="py-2 px-3 text-center text-slate-300">${item.count}</td>
                        <td class="py-2 px-3 text-right text-slate-400 font-mono">${(item.log_likelihood_ratio > 0 ? '+' : '')}${item.log_likelihood_ratio.toFixed(3)}</td>
                        <td class="py-2 px-3 text-right ${colorCls} font-bold font-mono">${(oppImpact > 0 ? '+' : '')}${oppImpact.toFixed(3)}</td>
                        <td class="py-2 px-3 text-[11px] text-slate-400">${share.toFixed(1)}%</td>
                    `;
                    tbodyCounter.appendChild(tr);
                });
            }
        }

        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        window.onload = () => {
            initPresets();
            // Load first preset by default
            if (PRESETS.length > 0) {
                loadPreset(PRESETS[0]);
            }
        };
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    import json
    return render_template_string(
        HTML_TEMPLATE,
        vocab_size=model.vocab_size,
        prior_log_odds=model.prior_log_odds,
        presets_json=json.dumps(CURATED_BENCHMARK_EMAILS[:6])
    )


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(force=True)
    text = data.get("text", "")
    top_k = int(data.get("top_k", 10))
    explanation = model.explain(text, top_k=top_k)
    return jsonify(explanation)


@app.route("/api/top-features", methods=["GET"])
def api_top_features():
    top_n = int(request.args.get("n", 20))
    features = model.get_top_features(top_n=top_n)
    return jsonify(features)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Naive Bayes Spam Classifier Web App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

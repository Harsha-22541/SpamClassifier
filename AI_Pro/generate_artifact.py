"""Generate standalone self-contained artifact widget from template."""

import json
import os
from src.dataset import CURATED_BENCHMARK_EMAILS

def build_widget():
    with open('models/naive_bayes_model.json', 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    llr_dict = model_data['log_likelihood_ratio']
    sorted_by_impact = sorted(llr_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    filtered_llr = {k: round(v, 4) for k, v in sorted_by_impact[:3500]}

    presets = CURATED_BENCHMARK_EMAILS[:8]
    prior = round(model_data['prior_log_odds'], 4)

    with open('templates/widget.html', 'r', encoding='utf-8') as f:
        template = f.read()

    rendered = template.replace('__LLR_JSON__', json.dumps(filtered_llr))
    rendered = rendered.replace('__PRIOR_LOG_ODDS__', str(prior))
    rendered = rendered.replace('__PRESETS_JSON__', json.dumps(presets))

    artifact_dir = r"C:\Users\hv728\.gemini\antigravity\brain\c1f6f6e0-e2c6-497a-afcf-796cb765a1f9"
    os.makedirs(artifact_dir, exist_ok=True)
    out_file = os.path.join(artifact_dir, "spam_classifier_widget.html")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Artifact successfully generated at: {out_file} ({len(rendered)} bytes)")

if __name__ == "__main__":
    build_widget()

"""Explainability and feature attribution reporting for Naive Bayes classification."""

import html
from typing import Dict, Any, List


class DecisionExplainer:
    """Format and visualize word influence reports for Naive Bayes classifications."""

    # ANSI color codes for terminal display
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BG_RED = "\033[41m\033[37m"
    BG_GREEN = "\033[42m\033[30m"

    @classmethod
    def format_cli_report(cls, explanation: Dict[str, Any], show_tokens: bool = True) -> str:
        """Create a detailed, beautiful terminal report explaining the classification decision."""
        pred = explanation['predicted_class'].upper()
        prob_spam = explanation['probabilities']['spam'] * 100.0
        prob_ham = explanation['probabilities']['ham'] * 100.0
        delta = explanation['total_log_odds']
        prior = explanation['prior_log_odds']
        net_words = explanation['net_word_evidence']

        is_spam = explanation['predicted_class'] == 'spam'
        badge_color = cls.RED if is_spam else cls.GREEN
        target_name = "SPAM" if is_spam else "HAM"

        lines = []
        lines.append("=" * 78)
        lines.append(f"{cls.BOLD}NAIVE BAYES CLASSIFICATION EXPLANATION REPORT{cls.RESET}".center(86))
        lines.append("=" * 78)

        # Decision Summary
        lines.append(f"\n{cls.BOLD}CLASSIFICATION RESULT:{cls.RESET}")
        lines.append(f"  Prediction:          {badge_color}{cls.BOLD}[ {pred} ]{cls.RESET}")
        lines.append(f"  Confidence:          {cls.BOLD}{explanation['confidence'] * 100.0:.2f}%{cls.RESET}")
        lines.append(f"  Spam Probability:    {cls.RED}{prob_spam:6.2f}%{cls.RESET}")
        lines.append(f"  Ham Probability:     {cls.GREEN}{prob_ham:6.2f}%{cls.RESET}")

        # Mathematical log-odds decomposition
        lines.append(f"\n{cls.BOLD}MATHEMATICAL LOG-ODDS DECOMPOSITION:{cls.RESET}")
        lines.append(f"  Formula: log-odds = Prior + Net Word Evidence")
        lines.append(f"  Prior Log-Odds (b):            {prior:+8.4f}  (base rate bias)")
        lines.append(f"  Net Word Log-Odds Sum (sum phi):{net_words:+8.4f}  (from email tokens)")
        lines.append(f"  -------------------------------------------")
        lines.append(f"  Total Posterior Log-Odds (Delta):{badge_color}{cls.BOLD}{delta:+8.4f}{cls.RESET}")
        lines.append(f"  Decision Threshold:            Delta > 0 => SPAM | Delta <= 0 => HAM")

        # Top words driving the decision
        top_supporting = explanation['top_influential_words']
        lines.append(f"\n{cls.BOLD}TOP WORDS INFLUENCING DECISION TOWARDS [{target_name}]:{cls.RESET}")
        if not top_supporting:
            lines.append("  (No vocabulary words found favoring this class)")
        else:
            lines.append(f"  {'Rank':<5} {'Word':<18} {'Count':<6} {'LLR (Weight)':<14} {'Net Influence':<16} {'Evidence Share'}")
            lines.append("  " + "-" * 74)
            for idx, item in enumerate(top_supporting, 1):
                word = item['word']
                cnt = item['count']
                llr = item['log_likelihood_ratio']
                impact = item['contribution_to_spam'] if is_spam else -item['contribution_to_spam']
                share = item.get('evidence_share', 0.0)

                # Bar for visualization
                bar_len = int(min(share / 4, 15))
                bar_str = "#" * max(bar_len, 1)

                val_color = cls.RED if is_spam else cls.GREEN
                lines.append(f"  #{idx:<4} {cls.BOLD}{word:<18}{cls.RESET} {cnt:<6} {llr:+12.3f}   {val_color}{impact:+12.3f}{cls.RESET}    {share:5.1f}% {val_color}{bar_str}{cls.RESET}")

        # Counter-evidence words (words pulling the other way)
        counter_words = explanation['top_opposing_words']
        opp_name = "HAM" if is_spam else "SPAM"
        lines.append(f"\n{cls.BOLD}COUNTER-EVIDENCE (WORDS PULLING TOWARDS [{opp_name}]):{cls.RESET}")
        if not counter_words:
            lines.append("  (None -- all present words agreed or were neutral)")
        else:
            lines.append(f"  {'Rank':<5} {'Word':<18} {'Count':<6} {'LLR (Weight)':<14} {'Opposing Impact':<16} {'Opposing Share'}")
            lines.append("  " + "-" * 74)
            for idx, item in enumerate(counter_words, 1):
                word = item['word']
                cnt = item['count']
                llr = item['log_likelihood_ratio']
                opp_impact = -item['contribution_to_spam'] if is_spam else item['contribution_to_spam']
                share = item.get('evidence_share', 0.0)

                val_color = cls.GREEN if is_spam else cls.RED
                lines.append(f"  #{idx:<4} {cls.BOLD}{word:<18}{cls.RESET} {cnt:<6} {llr:+12.3f}   {val_color}{opp_impact:+12.3f}{cls.RESET}    {share:5.1f}%")

        # OOV Words
        oov = explanation.get('oov_words', [])
        if oov:
            lines.append(f"\n{cls.GRAY}Out-of-Vocabulary words (neutral impact): {', '.join(oov[:15])}{'...' if len(oov) > 15 else ''}{cls.RESET}")

        lines.append("\n" + "=" * 78)
        return "\n".join(lines)

    @classmethod
    def generate_html_heatmap(cls, explanation: Dict[str, Any], title: str = "Email Word Influence Analysis") -> str:
        """Generate an interactive HTML heatmap visualizing per-word influence on classification."""
        pred = explanation['predicted_class'].upper()
        prob_spam = explanation['probabilities']['spam'] * 100.0
        prob_ham = explanation['probabilities']['ham'] * 100.0
        delta = explanation['total_log_odds']
        prior = explanation['prior_log_odds']
        net_words = explanation['net_word_evidence']
        is_spam = explanation['predicted_class'] == 'spam'

        # Map words to contribution
        word_map = {item['word']: item for item in explanation['all_words_breakdown']}

        # Color token generation
        raw_text = explanation['text']
        # Highlight words in the text
        import re
        tokens_and_delims = re.split(r'(\b[a-zA-Z_][a-zA-Z0-9_]*\b)', raw_text)

        html_snippets = []
        for segment in tokens_and_delims:
            seg_lower = segment.lower()
            if seg_lower in word_map:
                w_info = word_map[seg_lower]
                llr = w_info['log_likelihood_ratio']
                contrib = w_info['contribution_to_spam']

                # Normalize intensity [0.15, 0.85]
                intensity = min(abs(llr) / 4.0, 1.0) * 0.7 + 0.15
                if contrib > 0.05:
                    # Red for Spam
                    bg_color = f"rgba(239, 68, 68, {intensity:.2f})"
                    txt_color = "#991b1b" if intensity < 0.5 else "#ffffff"
                    tooltip = f"Word: &apos;{html.escape(segment)}&apos; | LLR: +{llr:.2f} (Spam-leaning) | Total: +{contrib:.2f}"
                elif contrib < -0.05:
                    # Green for Ham
                    bg_color = f"rgba(34, 197, 94, {intensity:.2f})"
                    txt_color = "#166534" if intensity < 0.5 else "#ffffff"
                    tooltip = f"Word: &apos;{html.escape(segment)}&apos; | LLR: {llr:.2f} (Ham-leaning) | Total: {contrib:.2f}"
                else:
                    bg_color = "transparent"
                    txt_color = "inherit"
                    tooltip = f"Word: &apos;{html.escape(segment)}&apos; | Neutral"

                snippet = f'<span style="background-color: {bg_color}; color: {txt_color}; padding: 1px 4px; border-radius: 4px; font-weight: 500; cursor: help;" title="{tooltip}">{html.escape(segment)}</span>'
                html_snippets.append(snippet)
            else:
                html_snippets.append(html.escape(segment))

        highlighted_email = "".join(html_snippets).replace("\n", "<br>")

        # Rows for top influential table
        rows_html = []
        for idx, item in enumerate(explanation['top_influential_words'][:10], 1):
            w = html.escape(item['word'])
            cnt = item['count']
            llr = item['log_likelihood_ratio']
            contrib = item['contribution_to_spam']
            share = item.get('evidence_share', 0.0)
            color_class = "text-red-500 font-semibold" if is_spam else "text-emerald-500 font-semibold"
            bar_color = "bg-red-500" if is_spam else "bg-emerald-500"

            rows_html.append(f"""
            <tr class="border-b border-[var(--border)] text-sm">
                <td class="py-2 px-3 text-[var(--muted-foreground)]">#{idx}</td>
                <td class="py-2 px-3 font-medium">{w}</td>
                <td class="py-2 px-3 text-center">{cnt}</td>
                <td class="py-2 px-3 text-right font-mono">{llr:+.3f}</td>
                <td class="py-2 px-3 text-right font-mono {color_class}">{contrib:+.3f}</td>
                <td class="py-2 px-3">
                    <div class="flex items-center gap-2">
                        <div class="w-16 bg-[var(--border)] h-2 rounded-full overflow-hidden">
                            <div class="{bar_color} h-full" style="width: {min(share, 100):.1f}%"></div>
                        </div>
                        <span class="text-xs text-[var(--muted-foreground)]">{share:.1f}%</span>
                    </div>
                </td>
            </tr>
            """)

        top_table_content = "\n".join(rows_html)

        badge_bg = "bg-red-500/20 text-red-400 border-red-500/40" if is_spam else "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(title)}</title>
    <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-4">
    <div class="max-w-4xl mx-auto bg-[var(--card)] border border-[var(--border)] rounded-2xl p-6 shadow-lg">
        <!-- Header -->
        <div class="flex items-center justify-between border-b border-[var(--border)] pb-4 mb-6">
            <div>
                <h1 class="text-xl font-bold text-[var(--foreground)]">{html.escape(title)}</h1>
                <p class="text-sm text-[var(--muted-foreground)]">Naive Bayes Exact Log-Odds Decision Decomposition</p>
            </div>
            <div class="flex items-center gap-3">
                <span class="px-4 py-1.5 rounded-full border text-sm font-bold {badge_bg}">
                    {pred} ({explanation['confidence'] * 100:.1f}%)
                </span>
            </div>
        </div>

        <!-- Metrics cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
                <div class="text-xs text-[var(--muted-foreground)] font-medium">Spam Probability</div>
                <div class="text-2xl font-bold text-red-500 mt-1">{prob_spam:.1f}%</div>
                <div class="text-xs text-[var(--muted-foreground)] mt-1">Ham: {prob_ham:.1f}%</div>
            </div>
            <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
                <div class="text-xs text-[var(--muted-foreground)] font-medium">Prior Log-Odds (b)</div>
                <div class="text-2xl font-bold font-mono text-[var(--foreground)] mt-1">{prior:+.3f}</div>
                <div class="text-xs text-[var(--muted-foreground)] mt-1">Class baseline bias</div>
            </div>
            <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
                <div class="text-xs text-[var(--muted-foreground)] font-medium">Word Evidence (Σ φ)</div>
                <div class="text-2xl font-bold font-mono {'text-red-500' if net_words > 0 else 'text-emerald-500'} mt-1">{net_words:+.3f}</div>
                <div class="text-xs text-[var(--muted-foreground)] mt-1">Tokens' log-odds sum</div>
            </div>
            <div class="p-4 rounded-xl bg-[var(--background)] border border-[var(--border)]">
                <div class="text-xs text-[var(--muted-foreground)] font-medium">Posterior Log-Odds (Δ)</div>
                <div class="text-2xl font-bold font-mono {'text-red-500' if delta > 0 else 'text-emerald-500'} mt-1">{delta:+.3f}</div>
                <div class="text-xs text-[var(--muted-foreground)] mt-1">> 0 => Spam, < 0 => Ham</div>
            </div>
        </div>

        <!-- Highlighted Email Heatmap -->
        <div class="mb-6">
            <div class="flex items-center justify-between mb-2">
                <h2 class="text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                    Email Token Influence Heatmap
                </h2>
                <div class="flex items-center gap-3 text-xs">
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-red-500 inline-block"></span> Spam Evidence</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-500 inline-block"></span> Ham Evidence</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-slate-400 inline-block"></span> Neutral / OOV</span>
                </div>
            </div>
            <div class="p-5 rounded-xl bg-[var(--background)] border border-[var(--border)] font-mono text-sm leading-relaxed max-h-72 overflow-y-auto">
                {highlighted_email}
            </div>
        </div>

        <!-- Top Driving Words Table -->
        <div class="mb-2">
            <h2 class="text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
                Top Words Driving This Decision ({'Spam' if is_spam else 'Ham'})
            </h2>
            <div class="overflow-x-auto border border-[var(--border)] rounded-xl">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="border-b border-[var(--border)] bg-[var(--background)] text-xs uppercase text-[var(--muted-foreground)]">
                            <th class="py-2.5 px-3">Rank</th>
                            <th class="py-2.5 px-3">Word</th>
                            <th class="py-2.5 px-3 text-center">Freq</th>
                            <th class="py-2.5 px-3 text-right">LLR (Weight)</th>
                            <th class="py-2.5 px-3 text-right">Total Influence</th>
                            <th class="py-2.5 px-3">Share of Evidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {top_table_content}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""

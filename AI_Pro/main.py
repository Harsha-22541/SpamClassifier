"""Command-Line Interface (CLI) for Naive Bayes Spam/Ham Classifier with Word Influence Reporting."""

import argparse
import sys
import os
from src.naive_bayes import NaiveBayesClassifier
from src.preprocessor import TextPreprocessor
from src.explain import DecisionExplainer
from src.dataset import load_sms_dataset, create_curated_email_dataset, train_test_split, CURATED_BENCHMARK_EMAILS
from src.evaluate import calculate_metrics, format_metrics_report, benchmark_against_sklearn

DEFAULT_MODEL_PATH = "models/naive_bayes_model.json"


def get_or_train_model(model_path: str = DEFAULT_MODEL_PATH, force_retrain: bool = False) -> NaiveBayesClassifier:
    """Load cached model or train a new one if not found."""
    preprocessor = TextPreprocessor(lowercase=True, normalize_urls=True, normalize_currency=True)

    if os.path.exists(model_path) and not force_retrain:
        try:
            return NaiveBayesClassifier.load(model_path, preprocessor=preprocessor)
        except Exception as e:
            print(f"[Warning] Failed to load model ({e}), retraining...")

    print("[*] Training Naive Bayes model on SMS and Curated Email datasets...")
    sms_texts, sms_labels = load_sms_dataset("data/sms_dataset.tsv")
    email_texts, email_labels = create_curated_email_dataset("data/emails.csv")

    all_texts = sms_texts + email_texts
    all_labels = sms_labels + email_labels

    classifier = NaiveBayesClassifier(alpha=1.0, preprocessor=preprocessor)
    classifier.fit(all_texts, all_labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save(model_path)
    print(f"[*] Model trained on {len(all_texts)} examples and saved to {model_path}\n")
    return classifier


def analyze_text(classifier: NaiveBayesClassifier, text: str, top_k: int = 10, export_html: str = None):
    """Classify text and print full word influence report."""
    explanation = classifier.explain(text, top_k=top_k)
    print(DecisionExplainer.format_cli_report(explanation))

    if export_html:
        html_content = DecisionExplainer.generate_html_heatmap(explanation, title="Custom Email Analysis")
        with open(export_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n[+] Interactive HTML Heatmap report exported to: {export_html}")


def run_interactive_mode(classifier: NaiveBayesClassifier, top_k: int = 10):
    """Interactive CLI shell for evaluating user emails in real time."""
    print("=" * 76)
    print(" INTERACTIVE NAIVE BAYES SPAM CLASSIFIER & WORD INFLUENCE ANALYZER")
    print(" Enter or paste email text below. (Type 'exit' or 'quit' to stop)")
    print("=" * 76)

    sample_prompts = [
        ("1", "Phishing Alert Scam", "URGENT: Your bank account access has been restricted. Click here to verify identity."),
        ("2", "Lottery Win Scam", "Congratulations! You won $1,000,000 cash in the international sweepstakes. Claim now!"),
        ("3", "Sprint Work Meeting", "Hi team, please review the attached slides before tomorrow's sprint meeting. Thanks!"),
        ("4", "Borderline Tech Event", "Claim your free ticket now to the software engineering conference next week.")
    ]

    while True:
        print("\nOptions:")
        print("  - Paste custom text and press [Enter]")
        print("  - Type 'samples' to choose a preloaded email")
        print("  - Type 'quit' to exit")

        try:
            user_input = input("\nEmail text > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'samples':
            print("\nPreloaded sample emails:")
            for key, name, preview in sample_prompts:
                print(f"  [{key}] {name}: \"{preview[:60]}...\"")
            choice = input("Select sample (1-4): ").strip()
            matched = next((item for item in sample_prompts if item[0] == choice), None)
            if matched:
                user_input = matched[2]
            else:
                print("Invalid choice, returning to prompt.")
                continue

        analyze_text(classifier, user_input, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(
        description="Naive Bayes Text Classifier for Spam/Ham Email Detection with Word Influence Reporting"
    )
    parser.add_argument("--classify", "-c", type=str, help="Email text to classify and explain")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive evaluation shell")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Evaluate model accuracy and benchmark against scikit-learn")
    parser.add_argument("--top-features", "-f", action="store_true", help="Show global top spam and ham indicator words")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining on datasets")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top influential words to display (default: 10)")
    parser.add_argument("--export-html", type=str, default=None, help="Path to export HTML heatmap report")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing parameter alpha (default: 1.0)")

    args = parser.parse_args()

    model = get_or_train_model(force_retrain=args.retrain)

    if args.evaluate:
        sms_texts, sms_labels = load_sms_dataset("data/sms_dataset.tsv")
        email_texts, email_labels = create_curated_email_dataset("data/emails.csv")
        all_texts = sms_texts + email_texts
        all_labels = sms_labels + email_labels

        train_texts, train_labels, test_texts, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_seed=42, stratify=True
        )
        test_preds = model.predict(test_texts)
        metrics = calculate_metrics(test_labels, test_preds)
        print(format_metrics_report(metrics))

        sk_res = benchmark_against_sklearn(train_texts, train_labels, test_texts, test_labels, model)
        if sk_res.get('status') == 'success':
            print(f"\nAgreement with scikit-learn MultinomialNB: {sk_res['prediction_agreement_rate'] * 100:.2f}%")
        return

    if args.top_features:
        features = model.get_top_features(top_n=args.top_k)
        print("=" * 64)
        print("GLOBAL TOP SPAM INDICATORS (Highest Log-Likelihood Ratios):")
        print("=" * 64)
        print(f"  {'Rank':<5} {'Token':<18} {'LLR':<10} {'Spam Count':<12} {'Ham Count'}")
        print("  " + "-" * 56)
        for idx, f in enumerate(features['top_spam_words'], 1):
            print(f"  #{idx:<4} {f['word']:<18} {f['log_likelihood_ratio']:+8.3f}   {f['spam_count']:<12} {f['ham_count']}")

        print("\n" + "=" * 64)
        print("GLOBAL TOP HAM INDICATORS (Lowest Log-Likelihood Ratios):")
        print("=" * 64)
        print(f"  {'Rank':<5} {'Token':<18} {'LLR':<10} {'Ham Count':<12} {'Spam Count'}")
        print("  " + "-" * 56)
        for idx, f in enumerate(features['top_ham_words'], 1):
            print(f"  #{idx:<4} {f['word']:<18} {f['log_likelihood_ratio']:+8.3f}   {f['ham_count']:<12} {f['spam_count']}")
        return

    if args.classify:
        analyze_text(model, args.classify, top_k=args.top_k, export_html=args.export_html)
        return

    if args.interactive:
        run_interactive_mode(model, top_k=args.top_k)
        return

    # Default action if no args provided: show top features and an example classification
    print("[*] No flag passed. Running sample classification. Use --help to view options.\n")
    sample_text = "URGENT: You won $5,000 cash! Claim your free gift prize right now at http://win-bonus.com"
    print(f"Classifying sample: \"{sample_text}\"\n")
    analyze_text(model, sample_text, top_k=args.top_k, export_html="reports/sample_report.html")


if __name__ == "__main__":
    main()

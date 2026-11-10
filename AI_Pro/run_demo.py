"""End-to-end training, evaluation, and demonstration of Naive Bayes email spam classifier with word influence reporting."""

import os
from src.dataset import load_sms_dataset, create_curated_email_dataset, train_test_split, CURATED_BENCHMARK_EMAILS
from src.preprocessor import TextPreprocessor
from src.naive_bayes import NaiveBayesClassifier
from src.explain import DecisionExplainer
from src.evaluate import calculate_metrics, format_metrics_report, benchmark_against_sklearn


def run_full_pipeline():
    print("=" * 80)
    print("NAIVE BAYES EMAIL SPAM/HAM CLASSIFIER WITH WORD INFLUENCE REPORTING")
    print("=" * 80)

    # 1. Prepare Datasets
    print("\n[1/5] Loading datasets...")
    sms_path = "data/sms_dataset.tsv"
    email_path = "data/emails.csv"

    sms_texts, sms_labels = load_sms_dataset(sms_path)
    curated_texts, curated_labels = create_curated_email_dataset(email_path)

    print(f"  Loaded {len(sms_texts)} SMS/message records ({sms_labels.count('ham')} ham, {sms_labels.count('spam')} spam)")
    print(f"  Created {len(curated_texts)} curated email records ({curated_labels.count('ham')} ham, {curated_labels.count('spam')} spam)")

    # Combine for rich vocabulary (emails + SMS)
    all_texts = sms_texts + curated_texts
    all_labels = sms_labels + curated_labels

    train_texts, train_labels, test_texts, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_seed=42, stratify=True
    )
    print(f"  Training set: {len(train_texts)} samples | Testing set: {len(test_texts)} samples")

    # 2. Train Naive Bayes Classifier
    print("\n[2/5] Training Multinomial Naive Bayes Classifier with Laplace Smoothing (alpha=1.0)...")
    preprocessor = TextPreprocessor(lowercase=True, normalize_urls=True, normalize_currency=True)
    classifier = NaiveBayesClassifier(alpha=1.0, preprocessor=preprocessor)
    classifier.fit(train_texts, train_labels)

    print(f"  Model trained successfully!")
    print(f"  Vocabulary size:      {classifier.vocab_size} unique tokens")
    print(f"  Ham tokens count:     {classifier.total_tokens['ham']}")
    print(f"  Spam tokens count:    {classifier.total_tokens['spam']}")
    print(f"  Prior Log-Odds (b):   {classifier.prior_log_odds:+.4f} (log P(spam)/P(ham))")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/naive_bayes_model.json"
    classifier.save(model_path)
    print(f"  Model serialized and saved to: {model_path}")

    # 3. Evaluate Classifier Performance
    print("\n[3/5] Evaluating test set performance...")
    test_preds = classifier.predict(test_texts)
    metrics = calculate_metrics(test_labels, test_preds)
    print(format_metrics_report(metrics, title="TEST SET EVALUATION RESULTS"))

    # Benchmark against scikit-learn
    print("\n[4/5] Benchmarking against scikit-learn's MultinomialNB...")
    sk_comparison = benchmark_against_sklearn(train_texts, train_labels, test_texts, test_labels, classifier)
    if sk_comparison.get('status') == 'success':
        agreed = sk_comparison['agreed_count']
        total = sk_comparison['total_tested']
        pct = sk_comparison['prediction_agreement_rate'] * 100.0
        print(f"  Agreement with scikit-learn MultinomialNB: {pct:.2f}% ({agreed}/{total} identical predictions)")
        print(f"  Our Scratch F1-Score:    {metrics['f1_score'] * 100:.2f}%")
        print(f"  Scikit-Learn F1-Score:   {sk_comparison['sklearn_metrics']['f1_score'] * 100:.2f}%")

    # 4. Global Top Informative Features
    print("\n[5/5] Global Top Feature Indicators:")
    top_features = classifier.get_top_features(top_n=10)
    print("\n  TOP 10 SPAM INDICATOR TOKENS (Highest Log-Likelihood Ratio):")
    print(f"    {'Rank':<5} {'Token':<18} {'LLR (Weight)':<14} {'Spam Count':<12} {'Ham Count'}")
    print("    " + "-" * 56)
    for idx, f in enumerate(top_features['top_spam_words'], 1):
        print(f"    #{idx:<4} {f['word']:<18} {f['log_likelihood_ratio']:+10.3f}    {f['spam_count']:<12} {f['ham_count']}")

    print("\n  TOP 10 HAM INDICATOR TOKENS (Lowest Log-Likelihood Ratio):")
    print(f"    {'Rank':<5} {'Token':<18} {'LLR (Weight)':<14} {'Ham Count':<12} {'Spam Count'}")
    print("    " + "-" * 56)
    for idx, f in enumerate(top_features['top_ham_words'], 1):
        print(f"    #{idx:<4} {f['word']:<18} {f['log_likelihood_ratio']:+10.3f}    {f['ham_count']:<12} {f['spam_count']}")

    # 5. Word Influence Explainability Demonstration on Diverse Emails
    print("\n" + "=" * 80)
    print("WORD INFLUENCE ATTRIBUTION DEMONSTRATION ON REALISTIC EMAILS")
    print("=" * 80)

    demo_cases = [
        CURATED_BENCHMARK_EMAILS[0],  # Phishing account suspension
        CURATED_BENCHMARK_EMAILS[1],  # Lottery win prize
        CURATED_BENCHMARK_EMAILS[5],  # Sprint planning meeting
        CURATED_BENCHMARK_EMAILS[6],  # Pull request code review
        CURATED_BENCHMARK_EMAILS[10]  # Borderline free tech conference ticket
    ]

    for sample in demo_cases:
        full_email = f"Subject: {sample['subject']}\n\n{sample['body']}"
        print(f"\nEvaluating: \"{sample['subject']}\" (Actual: {sample['label'].upper()} - {sample['notes']})")
        explanation = classifier.explain(full_email, top_k=7)
        report_str = DecisionExplainer.format_cli_report(explanation)
        print(report_str)

        # Generate HTML heatmap
        os.makedirs("reports", exist_ok=True)
        html_filename = f"reports/{sample['id']}_influence.html"
        html_content = DecisionExplainer.generate_html_heatmap(explanation, title=sample['subject'])
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"  [Saved HTML Heatmap Report]: {html_filename}")


if __name__ == "__main__":
    run_full_pipeline()

"""Multinomial Naive Bayes text classifier with exact log-odds word influence attribution."""

import math
import json
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional, Union
from src.preprocessor import TextPreprocessor


class NaiveBayesClassifier:
    """Multinomial Naive Bayes text classifier with exact word influence attribution.
    
    Mathematical Formulation:
        Let C in {spam, ham}.
        Log posterior odds:
            log [ P(spam | d) / P(ham | d) ] = log [ P(spam) / P(ham) ]
                                             + sum_{w in d} count(w, d) * [ log P(w | spam) - log P(w | ham) ]
        
        Word influence for word w in document d:
            phi(w, d) = count(w, d) * LLR(w)
            where LLR(w) = log P(w | spam) - log P(w | ham)
            
        A positive phi(w, d) pushes the decision towards SPAM.
        A negative phi(w, d) pushes the decision towards HAM.
    """

    def __init__(self, alpha: float = 1.0, preprocessor: Optional[TextPreprocessor] = None):
        """Initialize classifier with Laplace smoothing parameter alpha."""
        self.alpha = float(alpha)
        self.preprocessor = preprocessor if preprocessor is not None else TextPreprocessor()

        # Classes
        self.classes: List[str] = ['ham', 'spam']
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.doc_count: int = 0

        # Word statistics per class
        # word_token_counts[class][word] = total times word appeared across all docs in class
        self.word_token_counts: Dict[str, Dict[str, int]] = {
            'ham': defaultdict(int),
            'spam': defaultdict(int)
        }
        # Total words (tokens) in each class
        self.total_tokens: Dict[str, int] = {'ham': 0, 'spam': 0}

        # Vocabulary
        self.vocab: Dict[str, int] = {}
        self.vocab_size: int = 0

        # Precomputed probabilities (log space)
        self.log_priors: Dict[str, float] = {}
        self.feature_log_prob: Dict[str, Dict[str, float]] = {'ham': {}, 'spam': {}}
        self.log_likelihood_ratio: Dict[str, float] = {}  # LLR(w) = log P(w|spam) - log P(w|ham)
        self.prior_log_odds: float = 0.0  # log P(spam) - log P(ham)

        self.is_fitted: bool = False

    def fit(self, texts: List[str], labels: List[str]) -> 'NaiveBayesClassifier':
        """Train the classifier on a collection of texts and corresponding labels ('spam' or 'ham')."""
        if len(texts) != len(labels):
            raise ValueError(f"Texts ({len(texts)}) and labels ({len(labels)}) must have equal length.")

        self.doc_count = len(texts)
        self.class_counts = defaultdict(int)
        self.word_token_counts = {'ham': defaultdict(int), 'spam': defaultdict(int)}
        self.total_tokens = {'ham': 0, 'spam': 0}
        unique_vocab = set()

        # Count frequencies
        for text, label in zip(texts, labels):
            lbl = label.lower().strip()
            if lbl not in self.classes:
                raise ValueError(f"Unknown class '{lbl}'. Supported classes: {self.classes}")

            self.class_counts[lbl] += 1
            tokens = self.preprocessor.tokenize(text)
            for token in tokens:
                unique_vocab.add(token)
                self.word_token_counts[lbl][token] += 1
                self.total_tokens[lbl] += 1

        # Build vocabulary mapping
        self.vocab = {word: idx for idx, word in enumerate(sorted(unique_vocab))}
        self.vocab_size = len(self.vocab)

        # Compute log priors with Laplace smoothing on class counts
        # P(c) = (count(c) + alpha) / (N + 2*alpha)
        total_priors_denom = self.doc_count + len(self.classes) * self.alpha
        for c in self.classes:
            self.log_priors[c] = math.log((self.class_counts[c] + self.alpha) / total_priors_denom)

        self.prior_log_odds = self.log_priors['spam'] - self.log_priors['ham']

        # Compute feature log probabilities with Laplace smoothing:
        # P(w | c) = (count(w, c) + alpha) / (total_tokens[c] + alpha * |V|)
        self.feature_log_prob = {'ham': {}, 'spam': {}}
        self.log_likelihood_ratio = {}

        denom_ham = self.total_tokens['ham'] + self.alpha * self.vocab_size
        denom_spam = self.total_tokens['spam'] + self.alpha * self.vocab_size

        for word in self.vocab:
            p_w_ham = (self.word_token_counts['ham'][word] + self.alpha) / denom_ham
            p_w_spam = (self.word_token_counts['spam'][word] + self.alpha) / denom_spam

            log_p_ham = math.log(p_w_ham)
            log_p_spam = math.log(p_w_spam)

            self.feature_log_prob['ham'][word] = log_p_ham
            self.feature_log_prob['spam'][word] = log_p_spam
            self.log_likelihood_ratio[word] = log_p_spam - log_p_ham

        self.is_fitted = True
        return self

    def _score_tokens(self, tokens: List[str]) -> Tuple[float, float, Dict[str, int], List[str]]:
        """Calculate log posteriors (unnormalized) and token breakdown for a token list."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        token_counts = Counter(tokens)
        log_score_ham = self.log_priors['ham']
        log_score_spam = self.log_priors['spam']

        oov_words = []
        for word, count in token_counts.items():
            if word in self.vocab:
                log_score_ham += count * self.feature_log_prob['ham'][word]
                log_score_spam += count * self.feature_log_prob['spam'][word]
            else:
                oov_words.append(word)

        return log_score_ham, log_score_spam, token_counts, oov_words

    def predict_proba_single(self, text: str) -> Dict[str, float]:
        """Compute posterior probabilities P(spam|d) and P(ham|d) using numerically stable softmax."""
        tokens = self.preprocessor.tokenize(text)
        log_ham, log_spam, _, _ = self._score_tokens(tokens)

        # Stable softmax / sigmoid from log-odds
        delta = log_spam - log_ham  # log-odds of spam vs ham
        # P(spam) = 1 / (1 + exp(-delta))
        if delta > 700:
            p_spam = 1.0
        elif delta < -700:
            p_spam = 0.0
        else:
            p_spam = 1.0 / (1.0 + math.exp(-delta))

        p_ham = 1.0 - p_spam
        return {'ham': p_ham, 'spam': p_spam}

    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """Predict class label ('ham' or 'spam') for one or more texts."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        predictions = []
        for text in text_list:
            probs = self.predict_proba_single(text)
            pred = 'spam' if probs['spam'] >= 0.5 else 'ham'
            predictions.append(pred)

        return predictions[0] if is_single else predictions

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict probabilities for a list of texts."""
        return [self.predict_proba_single(t) for t in texts]

    def explain(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """Deconstruct the classification decision into exact word-level log-odds contributions.
        
        Returns:
            Dictionary containing:
            - predicted_class: 'spam' or 'ham'
            - probabilities: {'ham': float, 'spam': float}
            - prior_log_odds: log(P(spam)/P(ham))
            - net_word_evidence: sum of all word contributions
            - total_log_odds: prior_log_odds + net_word_evidence
            - top_influential_words: words pushing most strongly towards the PREDICTED class
            - top_opposing_words: words pushing towards the OPPOSING class (counter-evidence)
            - all_words_breakdown: detailed list of all recognized words and their weights
            - oov_words: words present in text but absent from training vocabulary
            - raw_tokens: token list in original order for heatmap visualization
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before explaining predictions.")

        raw_tokens = self.preprocessor.tokenize(text)
        log_ham, log_spam, token_counts, oov_words = self._score_tokens(raw_tokens)

        # Log odds: Delta = log(P(spam|d)/P(ham|d))
        delta = log_spam - log_ham
        if delta > 700:
            p_spam = 1.0
        elif delta < -700:
            p_spam = 0.0
        else:
            p_spam = 1.0 / (1.0 + math.exp(-delta))
        p_ham = 1.0 - p_spam

        predicted_class = 'spam' if p_spam >= 0.5 else 'ham'
        confidence = p_spam if predicted_class == 'spam' else p_ham

        # Analyze each word's contribution
        # phi(w) = count(w) * [log P(w|spam) - log P(w|ham)]
        word_breakdown = []
        net_word_evidence = 0.0

        for word, count in token_counts.items():
            if word in self.vocab:
                log_p_spam = self.feature_log_prob['spam'][word]
                log_p_ham = self.feature_log_prob['ham'][word]
                llr = self.log_likelihood_ratio[word]  # positive => spam, negative => ham
                contribution = count * llr
                net_word_evidence += contribution

                direction = 'spam' if contribution > 0 else ('ham' if contribution < 0 else 'neutral')
                word_breakdown.append({
                    'word': word,
                    'count': count,
                    'log_p_spam': log_p_spam,
                    'log_p_ham': log_p_ham,
                    'log_likelihood_ratio': llr,
                    'contribution_to_spam': contribution,
                    'abs_contribution': abs(contribution),
                    'direction': direction
                })

        # Separate words supporting the predicted class vs opposing it
        if predicted_class == 'spam':
            # Supporting words have positive contribution (towards spam)
            supporting_words = [w for w in word_breakdown if w['contribution_to_spam'] > 0]
            supporting_words.sort(key=lambda item: item['contribution_to_spam'], reverse=True)

            # Opposing words have negative contribution (mitigating spam)
            opposing_words = [w for w in word_breakdown if w['contribution_to_spam'] < 0]
            opposing_words.sort(key=lambda item: item['contribution_to_spam'])  # most negative first
        else:
            # Supporting words have negative contribution to spam => positive contribution to ham
            supporting_words = [w for w in word_breakdown if w['contribution_to_spam'] < 0]
            supporting_words.sort(key=lambda item: item['contribution_to_spam'])  # most negative first

            # Opposing words have positive contribution to spam => pushing away from ham
            opposing_words = [w for w in word_breakdown if w['contribution_to_spam'] > 0]
            opposing_words.sort(key=lambda item: item['contribution_to_spam'], reverse=True)

        # Calculate percentage share of total supporting evidence
        total_supporting_mag = sum(abs(w['contribution_to_spam']) for w in supporting_words)
        for w in supporting_words:
            w['evidence_share'] = (abs(w['contribution_to_spam']) / total_supporting_mag * 100.0) if total_supporting_mag > 0 else 0.0

        total_opposing_mag = sum(abs(w['contribution_to_spam']) for w in opposing_words)
        for w in opposing_words:
            w['evidence_share'] = (abs(w['contribution_to_spam']) / total_opposing_mag * 100.0) if total_opposing_mag > 0 else 0.0

        # Sort full breakdown by absolute influence
        word_breakdown.sort(key=lambda item: item['abs_contribution'], reverse=True)

        return {
            'text': text,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {'ham': p_ham, 'spam': p_spam},
            'prior_log_odds': self.prior_log_odds,
            'net_word_evidence': net_word_evidence,
            'total_log_odds': delta,
            'top_influential_words': supporting_words[:top_k],
            'top_opposing_words': opposing_words[:top_k],
            'all_words_breakdown': word_breakdown,
            'oov_words': sorted(list(set(oov_words))),
            'token_count': len(raw_tokens),
            'unique_vocab_tokens': len(word_breakdown)
        }

    def get_top_features(self, top_n: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Return the most informative spam indicators (highest LLR) and ham indicators (lowest LLR) globally."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before querying top features.")

        # LLR = log P(w|spam) - log P(w|ham)
        # Highest LLR => strongest Spam indicators
        # Lowest LLR => strongest Ham indicators
        sorted_features = sorted(self.log_likelihood_ratio.items(), key=lambda item: item[1])

        top_ham = []
        for word, llr in sorted_features[:top_n]:
            top_ham.append({
                'word': word,
                'log_likelihood_ratio': llr,
                'ham_strength': -llr,
                'ham_count': self.word_token_counts['ham'].get(word, 0),
                'spam_count': self.word_token_counts['spam'].get(word, 0)
            })

        top_spam = []
        for word, llr in reversed(sorted_features[-top_n:]):
            top_spam.append({
                'word': word,
                'log_likelihood_ratio': llr,
                'spam_strength': llr,
                'spam_count': self.word_token_counts['spam'].get(word, 0),
                'ham_count': self.word_token_counts['ham'].get(word, 0)
            })

        return {'top_spam_words': top_spam, 'top_ham_words': top_ham}

    def save(self, filepath: str) -> None:
        """Serialize model parameters to a JSON file."""
        data = {
            'alpha': self.alpha,
            'classes': self.classes,
            'class_counts': dict(self.class_counts),
            'doc_count': self.doc_count,
            'total_tokens': self.total_tokens,
            'vocab': self.vocab,
            'log_priors': self.log_priors,
            'prior_log_odds': self.prior_log_odds,
            'feature_log_prob': self.feature_log_prob,
            'log_likelihood_ratio': self.log_likelihood_ratio,
            'word_token_counts': {
                'ham': dict(self.word_token_counts['ham']),
                'spam': dict(self.word_token_counts['spam'])
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str, preprocessor: Optional[TextPreprocessor] = None) -> 'NaiveBayesClassifier':
        """Load serialized model parameters from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model = cls(alpha=data['alpha'], preprocessor=preprocessor)
        model.classes = data['classes']
        model.class_counts = defaultdict(int, data['class_counts'])
        model.doc_count = data['doc_count']
        model.total_tokens = data['total_tokens']
        model.vocab = data['vocab']
        model.vocab_size = len(model.vocab)
        model.log_priors = data['log_priors']
        model.prior_log_odds = data['prior_log_odds']
        model.feature_log_prob = data['feature_log_prob']
        model.log_likelihood_ratio = data['log_likelihood_ratio']
        model.word_token_counts = {
            'ham': defaultdict(int, data['word_token_counts']['ham']),
            'spam': defaultdict(int, data['word_token_counts']['spam'])
        }
        model.is_fitted = True
        return model

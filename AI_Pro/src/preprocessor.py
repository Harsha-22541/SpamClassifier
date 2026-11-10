"""Text preprocessing and tokenization for email spam/ham detection."""

import re
from typing import List, Set, Optional

# Standard English stopwords, carefully excluding words that are informative for spam/ham detection
DEFAULT_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t',
    'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t',
    'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
    'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve',
    'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more',
    'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she',
    'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that',
    'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
    'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve',
    'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while',
    'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you',
    'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

# Spam-sensitive words that should NEVER be filtered as stopwords
SPAM_SENSITIVE_WORDS = {
    'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'now', 'claim', 'guaranteed',
    'call', 'text', 'click', 'reply', 'offer', 'credit', 'buy', 'save', 'money', 'only'
}


class TextPreprocessor:
    """Configurable text preprocessor and tokenizer for spam/ham classification."""

    def __init__(
        self,
        lowercase: bool = True,
        normalize_urls: bool = True,
        normalize_emails: bool = True,
        normalize_currency: bool = True,
        normalize_numbers: bool = False,
        remove_stopwords: bool = False,
        min_word_len: int = 2,
        custom_stopwords: Optional[Set[str]] = None
    ):
        self.lowercase = lowercase
        self.normalize_urls = normalize_urls
        self.normalize_emails = normalize_emails
        self.normalize_currency = normalize_currency
        self.normalize_numbers = normalize_numbers
        self.remove_stopwords = remove_stopwords
        self.min_word_len = min_word_len

        if custom_stopwords is not None:
            self.stopwords = set(custom_stopwords)
        else:
            self.stopwords = DEFAULT_STOPWORDS - SPAM_SENSITIVE_WORDS

        # Regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.currency_pattern = re.compile(r'[\$£€¥](\d+(?:,\d+)*(?:\.\d+)?)')
        self.number_pattern = re.compile(r'\b\d+(?:,\d+)*(?:\.\d+)?\b')
        self.word_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')

    def clean_text(self, text: str) -> str:
        """Apply text normalization substitutions."""
        if not text or not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.normalize_urls:
            text = self.url_pattern.sub(' __url__ ', text)

        if self.normalize_emails:
            text = self.email_pattern.sub(' __email__ ', text)

        if self.normalize_currency:
            text = self.currency_pattern.sub(r' __currency__ \1 ', text)

        if self.normalize_numbers:
            text = self.number_pattern.sub(' __number__ ', text)

        return text

    def tokenize(self, text: str) -> List[str]:
        """Convert raw email string into cleaned token list."""
        cleaned = self.clean_text(text)
        tokens = self.word_pattern.findall(cleaned)

        filtered: List[str] = []
        for token in tokens:
            if len(token) < self.min_word_len:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            filtered.append(token)

        return filtered

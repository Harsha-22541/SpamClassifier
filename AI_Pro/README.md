# Naive Bayes Spam/Ham Classifier with Word Influence Attribution

A high-performance **Multinomial Naive Bayes** email classification system with **exact log-odds feature attribution** that explains *why* each email was classified as Spam or Ham and reports the most influential words driving each decision.

---

## 🌟 Key Features

1. **Exact Mathematical Decision Attribution**:
   Decomposes the posterior log-odds into the baseline class bias (prior log-odds) and the additive contribution of every individual token ($\phi(w, d) = \text{count}(w, d) \times \text{LLR}(w)$).
2. **Dual-Direction Evidence Reporting**:
   - **Drivers**: Words pushing most strongly towards the predicted class.
   - **Counter-Evidence**: Mitigating words pulling towards the opposing class.
   - **Evidence Share**: Percentage of the total decision weight accounted for by each word.
3. **High Accuracy & scikit-learn Parity**:
   - Accuracy: **98.22%**
   - Spam Precision: **92.90%**
   - Spam Recall: **94.12%**
   - Agreement with `scikit-learn.naive_bayes.MultinomialNB`: **99.47%**
4. **Rich Visualization & Reporting**:
   - Terminal colorized reporting with ASCII evidence bars.
   - Interactive HTML email token heatmap with hover tooltips showing individual word log-likelihood ratios.
   - Interactive Web Dashboard (Flask + Tailwind CSS) with real-time email scoring.
5. **Robust Preprocessor**:
   - Normalizes URLs (`__url__`), email addresses (`__email__`), and currency amounts (`__currency__`).
   - Retains spam-sensitive keywords (e.g. `free`, `win`, `prize`, `claim`, `urgent`, `cash`).

---

## 📐 Mathematical Formulation

### 1. Generative Naive Bayes Model
Given an email document $d$ represented by token counts $x = (x_1, x_2, \dots, x_{|V|})$, Bayes' theorem gives the posterior probability for class $c \in \{\text{Spam}, \text{Ham}\}$:

$$P(c \mid d) \propto P(c) \prod_{i=1}^{|V|} P(w_i \mid c)^{x_i}$$

Using Laplace smoothing ($\alpha = 1.0$):
$$\hat{P}(c) = \frac{N_c + \alpha}{N + 2\alpha}$$
$$\hat{P}(w_i \mid c) = \frac{\text{count}(w_i, c) + \alpha}{\sum_{w \in V} \text{count}(w, c) + \alpha |V|}$$

### 2. Log-Odds & Decision Attribution
Taking the ratio of log posteriors yields the **log-odds**:

$$\Delta(d) = \log \frac{P(\text{Spam} \mid d)}{P(\text{Ham} \mid d)} = \log \frac{P(\text{Spam})}{P(\text{Ham})} + \sum_{w_i \in d} x_i \cdot \underbrace{\left[ \log P(w_i \mid \text{Spam}) - \log P(w_i \mid \text{Ham}) \right]}_{\text{LLR}(w_i)}$$

$$\Delta(d) = b + \sum_{w_i \in d} \phi(w_i, d)$$

where:
- $b = \log \frac{P(\text{Spam})}{P(\text{Ham})}$ is the **Prior Log-Odds** (dataset base rate bias).
- $\text{LLR}(w_i)$ is the **Log-Likelihood Ratio** (weight of word $w_i$).
  - $\text{LLR}(w_i) > 0 \implies$ Word is evidence for **Spam**.
  - $\text{LLR}(w_i) < 0 \implies$ Word is evidence for **Ham**.
- $\phi(w_i, d) = x_i \cdot \text{LLR}(w_i)$ is the **exact additive contribution** of word $w_i$ to document $d$.
- If $\Delta(d) > 0 \implies$ Classified as **Spam** ($P(\text{Spam} \mid d) > 0.5$).
- If $\Delta(d) \le 0 \implies$ Classified as **Ham** ($P(\text{Ham} \mid d) \ge 0.5$).

### 3. Posterior Probability Conversion
Using the logistic sigmoid function $\sigma$:
$$P(\text{Spam} \mid d) = \frac{1}{1 + e^{-\Delta(d)}}, \quad P(\text{Ham} \mid d) = 1 - P(\text{Spam} \mid d)$$

---

## 🚀 Quickstart

### 1. Run Complete Demonstration
Train on combined SMS & Curated Email datasets, run evaluation, benchmark against scikit-learn, and print word influence reports:
```bash
python run_demo.py
```

### 2. Classify Any Custom Email via CLI
```bash
python main.py --classify "URGENT: Click here to verify your bank password and claim $1,000"
```
Or export an interactive HTML heatmap report:
```bash
python main.py --classify "Hi team, please review tomorrow's sprint agenda" --export-html report.html
```

### 3. Interactive Terminal Shell
```bash
python main.py --interactive
```

### 4. Launch Interactive Web Dashboard
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## 📊 Example CLI Explanation Output

```text
==============================================================================
                NAIVE BAYES CLASSIFICATION EXPLANATION REPORT                 
==============================================================================

CLASSIFICATION RESULT:
  Prediction:          [ SPAM ]
  Confidence:          100.00%
  Spam Probability:    100.00%
  Ham Probability:       0.00%

MATHEMATICAL LOG-ODDS DECOMPOSITION:
  Formula: log-odds = Prior + Net Word Evidence
  Prior Log-Odds (b):             -1.8486  (base rate bias)
  Net Word Log-Odds Sum (sum phi):+17.4392  (from email tokens)
  -------------------------------------------
  Total Posterior Log-Odds (Delta):+15.5906
  Decision Threshold:            Delta > 0 => SPAM | Delta <= 0 => HAM

TOP WORDS INFLUENCING DECISION TOWARDS [SPAM]:
  Rank  Word               Count  LLR (Weight)   Net Influence    Evidence Share
  --------------------------------------------------------------------------
  #1    claim              1            +5.636         +5.636     28.7% #######
  #2    prize              1            +5.402         +5.402     27.5% ######
  #3    cash               1            +3.518         +3.518     17.9% ####
  #4    won                1            +2.827         +2.827     14.4% ###
  #5    urgent             1            +2.235         +2.235     11.4% ##

COUNTER-EVIDENCE (WORDS PULLING TOWARDS [HAM]):
  Rank  Word               Count  LLR (Weight)   Opposing Impact  Opposing Share
  --------------------------------------------------------------------------
  #1    your               1            -0.640         +0.640     100.0%
==============================================================================
```

---

## 📂 Project Structure

```
AI_Pro/
├── data/
│   ├── sms_dataset.tsv            # 5,574 labeled SMS messages
│   └── emails.csv                 # Realistic curated business and personal emails
├── src/
│   ├── __init__.py
│   ├── preprocessor.py            # Tokenization, regex normalization, stopword filtering
│   ├── naive_bayes.py             # Multinomial NB model + exact log-odds decomposition
│   ├── dataset.py                 # Dataset loaders, stratified split, benchmark cases
│   ├── explain.py                 # Decision explainer, CLI formatter, HTML heatmap
│   └── evaluate.py                # Metrics (Precision, Recall, F1, FPR) & sklearn benchmark
├── models/
│   └── naive_bayes_model.json     # Serialized trained model
├── reports/                       # Generated HTML heatmap visual reports
├── tests/
│   └── test_naive_bayes.py        # Automated unit test suite
├── app.py                         # Interactive Flask Web Dashboard
├── main.py                        # CLI tool (classify, interactive, evaluate, top-features)
├── run_demo.py                    # End-to-end training and evaluation script
└── README.md                      # Documentation and mathematical guide
```

---

## 🧪 Running Unit Tests

```bash
python -m unittest discover tests
```
All unit tests verify:
- Laplace smoothing and probability computation
- Exact additivity of word influence ($\Delta = b + \sum \phi_i$)
- Out-of-vocabulary word handling
- Model JSON serialization and reload
- Precision, recall, and F1 calculations

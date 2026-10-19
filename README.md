# Market Reaction Fingerprint

### S&P Global & Crisil Campus Hackathon 2026

**Candidate Name:** Mehak Arora  
**College / Campus:** Indira Gandhi Delhi Technical University for Women (IGDTUW)  
**Module Alignment:** Module A — A tactical, high frequency stock index rebalancer. 
**Demo Video:** []  
**Presentation Deck:** [View Presentation](docs/presentation.pdf)  
**Architecture Diagram:** [View Architecture](docs/architecture.png)

**GitHub Repository:** https://github.com/mehakar09/igdtuw-mehak-arora-hackathon

---
## Live Demo

[Open Market Reaction Fingerprint](YOUR_DEPLOYED_STREAMLIT_URL)

## 1. Project Overview

Financial news can influence investor sentiment and stock prices, but different stocks may react differently to similar news signals. Market Reaction Fingerprint investigates whether stock-specific relationships exist between financial news sentiment and subsequent market returns, and whether these relationships provide useful information for portfolio decisions.

The application combines financial news sentiment analysis, historical stock-price data, statistical correlation analysis and portfolio backtesting. It uses FinBERT to estimate positive, negative and neutral sentiment probabilities for financial news and calculates a sentiment score as the positive probability minus the negative probability.

The system aggregates sentiment signals by trading date and stock, analyzes correlations between sentiment signals and future single-day returns at lags of 1, 2, 3 and 5 trading sessions, and evaluates a sentiment-based portfolio strategy against an equal-weight benchmark after accounting for transaction costs. A Streamlit dashboard presents the resulting stock analytics, reaction fingerprints and portfolio comparisons.

The project is aligned with the broader portfolio-analysis objective of Module B. It is an exploratory sentiment-analysis and backtesting prototype, not a claim of fully implemented portfolio stress testing.

## 2. Architecture and Tech Stack

### System Architecture

The application follows this processing pipeline:

1. **Data Ingestion:** Load financial news and historical stock-price data.
2. **Data Preparation:** Clean and align news and price records using available trading dates.
3. **Sentiment Analysis:** Use FinBERT to estimate positive, negative and neutral sentiment.
4. **Daily Aggregation:** Aggregate sentiment signals by stock and trading date.
5. **Reaction Fingerprint:** Calculate correlations between sentiment signals and future single-day stock returns at different trading-session lags.
6. **Portfolio Construction:** Convert sentiment signals into portfolio weights with position limits and smoothing.
7. **Backtesting:** Evaluate portfolio performance while accounting for transaction costs and compare it with an equal-weight benchmark.
8. **Visualization:** Display the outputs in an interactive Streamlit dashboard.

**Architecture Diagram:** [docs/architecture.png](docs/architecture.png)

### Technology Stack

- **Programming Language:** Python
- **NLP Model:** FinBERT — `ProsusAI/finbert`
- **NLP Framework:** Hugging Face Transformers
- **Data Processing:** Pandas, NumPy
- **Statistical Analysis:** Pearson correlation and false discovery rate (FDR) correction
- **Dashboard:** Streamlit
- **Data Storage:** CSV files
- **Visualization:** [List the plotting libraries actually used in the project]

## 3. Dataset Used

### Financial News Dataset

- **Dataset:** FNSPID NASDAQ financial news dataset
- **Source:** [FNSPID on Hugging Face](https://huggingface.co/datasets/sabareesh88/FNSPID_nasdaq)
- **Split:** Training split
- **Relevant fields:** Date, article title, stock symbol, publisher, author, URL and article text.
- **Purpose:** Financial news sentiment analysis using FinBERT.

### Historical Stock-Price Data

The analysis covers the following 11 stock symbols:

`NVDA, AAPL, GE, CAT, PG, KO, WMT, CVX, XOM, DIS, LLY`

**Analysis period:** January 2018 to January 2024.

Historical closing prices are used to calculate returns, analyze sentiment-return relationships and evaluate portfolio performance.

### Data Assumptions and Limitations

- News records are aligned with the available trading-date calendar.
- Future-return lags represent individual future trading-session returns, not cumulative returns over the entire lag period.
- Exact publication-time information is incomplete or unreliable for many records, so perfect intraday news-to-market alignment cannot be guaranteed.
- Correlation indicates association, not causation.
- Financial time-series dependence may affect the interpretation of conventional Pearson correlation significance tests.

The data sources and assumptions should be reviewed before interpreting the results. Only publicly available or otherwise appropriately licensed data is used; no confidential S&P Global or Crisil client data is used.

## 4. Quickstart and Installation

### Prerequisites

- Python
- Git
- Internet access to install dependencies and download the FinBERT model if it is not cached locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/mehakar09/igdtuw-mehak-arora-hackathon.git
cd igdtuw-mehak-arora-hackathon
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

**Note:** The commands above assume that `app.py` is located in the repository root and that all required input files are available at the paths expected by the code. Update these instructions if the actual project structure differs.

## 5. Key Results

### Reaction Fingerprint

- Evaluated 88 stock-signal-lag combinations in the full-sample analysis.
- No tested combination remained statistically significant after false discovery rate correction.
- A chronological train-test analysis was used to investigate the stability of the observed relationships.

These findings suggest that the tested sentiment-return relationships did not demonstrate robust statistical significance after correcting for multiple comparisons.

### Portfolio Backtest

The latest reported results are shown below.

| Metric | Sentiment Strategy | Equal-Weight Benchmark |
|---|---:|---:|
| Total return | -11.46% | +175.39% |
| Annualized volatility | 17.32% | 20.96% |
| Sharpe ratio | -0.031 | 0.913 |
| Maximum drawdown | -38.15% | -33.74% |
| Average daily turnover | 40.09% | 1.16% |
| Average cash allocation | 23.69% | 0% |

The sentiment-based strategy underperformed the equal-weight benchmark over the evaluated period. Although it exhibited lower annualized volatility, it produced a negative Sharpe ratio and a larger maximum drawdown.

These results demonstrate why sentiment signals must be evaluated using statistical validation, transaction costs and benchmark comparisons rather than assumed to provide a profitable trading advantage.

## 6. Domain Impact

Market Reaction Fingerprint provides an analytical framework for examining whether financial-news sentiment contains useful stock-specific information.

Potential applications include:

- Exploring how different stocks respond to financial news sentiment.
- Comparing sentiment signals across multiple trading-session horizons.
- Investigating the stability of observed relationships across time.
- Evaluating whether sentiment-derived portfolio weights improve on a simple benchmark.
- Identifying limitations before considering sentiment signals in a practical investment workflow.

The results are exploratory and do not establish that news sentiment predicts future returns or that the strategy is suitable for live trading.

## 7. Limitations and Future Work

- Improve publication-time accuracy and news-to-market alignment.
- Evaluate statistical significance using methods that account for time-series dependence.
- Test signals on additional unseen periods and datasets.
- Explore alternative portfolio construction methods, risk controls and transaction-cost assumptions.
- Extend the system with explicit event-driven portfolio stress scenarios to better address the full Module B use case.

## 8. Repository Structure

```text
igdtuw-mehak-arora-hackathon/
├── README.md
├── requirements.txt
├── LICENSE
├── app.py
├── src/
│   ├── daily_sentiment.py
│   ├── reaction_fingerprint.py
│   ├── portfolio.py
│   ├── backtest.py
│   └── strategy_evaluation.py
├── data/
│   ├── prices/
│   └── ...
└── docs/
    ├── presentation.pdf
    └── architecture.png
```

This structure is illustrative. The actual repository may contain additional files or use different locations. The documentation should match the committed project files.

## 9. Reproducibility and Disclaimer

This project was developed for educational and hackathon demonstration purposes. Historical performance does not guarantee future results. The analysis is not financial advice or a recommendation to buy or sell securities.

The dataset sources, preprocessing assumptions and analysis limitations are documented to support transparent interpretation of the results.

## 10. License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Project Links**

- [GitHub Repository](https://github.com/mehakar09/igdtuw-mehak-arora-hackathon)
- [Presentation Deck](docs/presentation.pdf)
- [Architecture Diagram](docs/architecture.png)
- Demo Video: 

# Cricket Momentum Score from Commentary

This project proposes a novel metric to quantify match momentum in cricket using ball-by-ball textual commentary data. Instead of relying solely on structured scorecard information (like runs or wickets), we use sentiment analysis to extract narrative signals from commentary that reflect the psychological ebb and flow of a match.

## Features

- **Commentary-Driven Momentum Index**: Sentiment score for each ball computed using VADER, adjusted with cricket-specific keyword matching.
- **Rolling Momentum Curve**: Smoothed using an 18-ball simple moving average.
- **Pressure Index Comparison**: Benchmarked against a traditional scorecard-based Pressure Index (PI).
- **Lightweight Implementation**: No deep learning models—built using pandas, nltk, and sklearn.

## Dataset

Source: [Kaggle - Cricket Scorecard and Commentary Dataset](https://www.kaggle.com/datasets/raghuvansht/cricket-scorecard-and-commentary-dataset)

## Usage
Run the script Momentum.py with a proper dataset like "1031665_COMMENTARY.csv"
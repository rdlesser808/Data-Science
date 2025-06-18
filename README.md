# Data Science

A collection of comprehensive data analysis projects showcasing various machine learning and statistical analysis techniques.

---

## Project 1: UK Weather Data Analysis (TADA_AE1.ipynb)

A comprehensive analysis of UK weather trends using Met Office historical station data from 1997-2022.

### Overview

This project examines weather patterns across 34 UK weather stations using data from the Met Office. The analysis focuses on temperature trends, rainfall patterns, and seasonal variations to identify climate patterns and anomalies over a 25-year period.

### Data Source

- **Provider**: UK Met Office
- **Stations**: 34 weather stations across the UK
- **Time Period**: 1997-2022
- **Data Link**: [Met Office Historic Station Data](https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data)

#### Excluded Stations
The following stations were excluded due to inconsistent data or closure:
- Cwmystwyth
- Ringway
- Southampton

### Key Features

#### Weather Metrics Analyzed
- Maximum Temperature (°C)
- Minimum Temperature (°C)
- Air Frost Days
- Rainfall (mm)
- Sunshine Duration (hours)
- Temperature Change Percentage

#### Visualizations
1. **Task 1**: Histogram charts showing average weather metrics by station
2. **Task 2**: Line graphs comparing temperature change trends for highest/lowest variance stations
3. **Task 3**: Box plots showing temperature distribution for top 10 warmest stations
4. **Task 4**: Comprehensive trend analysis with supporting charts

### Key Findings

#### Temperature Trends
- **Overall warming**: ~0.5°C increase from 1997-2022
- **Highest temperature change**: Yeovilton station
- **Lowest temperature change**: Aberporth station

#### Notable Anomalies
- **2010 Winter**: Significant temperature variance spike across all stations
- Ranked as 7th coldest winter in UK's last 100 years
- Consistent pattern observed across all weather stations

#### Climate Validation
- Results align with UK government temperature data
- Matches global temperature increase trends (NOAA: ~0.6°C globally)

---

## Project 2: Movie Review Sentiment Analysis (TADA_AE2.ipynb)

A machine learning project implementing sentiment analysis on movie reviews using the Stanford Sentiment Treebank (SST-2) dataset from the GLUE benchmark.

### Overview

This project develops and deploys a machine learning pipeline to classify movie review sentences as positive or negative sentiment. The analysis includes comprehensive data preprocessing, feature engineering, model selection, hyperparameter tuning, and ethical considerations.

### Data Source

- **Dataset**: Stanford Sentiment Treebank (SST-2) from GLUE benchmark
- **Size**: ~68,000 sentences from movie reviews
- **Labels**: Binary classification (0 = negative, 1 = positive)
- **Files**: train.tsv, dev.tsv, test.tsv
- **Source**: [GLUE Benchmark Tasks](https://gluebenchmark.com/tasks)

### Technical Pipeline

#### 1. Data Processing & Analysis
- **Class Distribution Analysis**: Training data contains 56% positive vs 44% negative reviews
- **Sentence Length Analysis**: Training data skewed toward shorter sentences (<15 words)
- **Dataset Imbalance**: Identified potential bias issues between training and testing distributions

#### 2. Text Vectorization (Ablation Study)
Tested four vectorization approaches:
- **TF-IDF** with and without stopwords
- **Bag-of-Words (CountVectorizer)** with and without stopwords
- **Selected Method**: Bag-of-Words with stopwords (best test performance)

#### 3. Machine Learning Model Selection
Compared two classification algorithms:
- **Logistic Regression** (liblinear solver)
- **Multinomial Naive Bayes**
- **Selected Model**: Logistic Regression (2% better performance across metrics)

#### 4. Hyperparameter Optimization
- **Method**: 5-fold Grid Search Cross-Validation
- **Parameters Tuned**: C (regularization), penalty (l1/l2), solver
- **Optimal Parameters**: C=0.7, penalty='l2', solver='liblinear'
- **Final Test Accuracy**: 82.22%

#### 5. Model Deployment
- **Serialization**: Models saved using joblib for production deployment
- **Evaluation**: Applied to unlabeled test set for real-world prediction simulation

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression (Optimized) | 82.22% | 82.3% | 82.2% | 82.2% |
| Multinomial Naive Bayes | 80.4% | 80.6% | 80.4% | 80.4% |

### Ethical Considerations

#### Identified Biases
- **Demographic Bias**: Dataset may reflect specific cultural/demographic perspectives
- **Genre Bias**: Model may favor certain movie genres or language patterns
- **Length Bias**: Training skewed toward shorter sentences may affect longer review classification

#### Mitigation Strategies
- Transparent methodology documentation
- Acknowledgment of dataset limitations
- Recommendation for diverse training data before production deployment

---

## Key Insights & Conclusions

### Weather Analysis
- **Climate Change Evidence**: Clear 0.5°C warming trend over 25 years
- **Extreme Weather**: 2010 winter anomaly affected all UK regions
- **Regional Variations**: Significant differences in temperature patterns across stations

### Sentiment Analysis
- **Model Performance**: 82.22% accuracy on balanced test data
- **Feature Engineering**: Bag-of-Words outperformed TF-IDF for this dataset
- **Bias Awareness**: Dataset imbalances require careful consideration for production use

## Future Improvements

### Weather Analysis
- Implement more sophisticated statistical methods
- Add confidence intervals and uncertainty quantification
- Develop predictive climate modeling
- Create interactive visualizations

### Sentiment Analysis
- Expand to multi-class sentiment classification
- Implement deep learning approaches (LSTM, BERT)
- Address dataset bias through data augmentation
- Deploy as web service with real-time prediction API

## Academic Context

Both projects were completed as part of the Theory and Applications of Data Analytics (TADA) course, demonstrating:
- **Statistical Analysis**: Descriptive statistics, trend analysis, hypothesis testing
- **Machine Learning**: Supervised learning, model selection, hyperparameter tuning
- **Data Visualization**: Clear, publication-ready charts and graphs
- **Ethical AI**: Consideration of bias, fairness, and societal impact
- **Scientific Rigor**: Reproducible methodology, proper citations, limitation acknowledgment

## References & Data Sources

### Weather Analysis
- UK Met Office Historical Station Data
- IPCC Special Report on 1.5°C
- UK Government Energy Trends
- NOAA Climate Data
- ResearchGate Publication on UK Winter Severity

### Sentiment Analysis
- GLUE Benchmark - Stanford Sentiment Treebank (SST-2)
- Scikit-learn Documentation
- IBM AI Ethics Resources
- OpenReview - Bias in Dataset Labels

## Repository Structure

```
repository/
├── README.md
├── TADA_AE1.ipynb    # UK Weather Data Analysis
└── TADA_AE2.ipynb    # Movie Review Sentiment Analysis
```

## License & Usage

This portfolio uses publicly available datasets and follows academic fair use guidelines. Weather data courtesy of UK Met Office; sentiment analysis data from Stanford NLP Group via GLUE benchmark. Please cite appropriately if using methodologies or findings from these analyses.

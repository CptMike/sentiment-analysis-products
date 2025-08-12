# Product Review Sentiment Analysis

The goal of this project is to analyze customer reviews to determine their sentiment—positive, negative, or neutral—based on the text content of the reviews and associated metadata. This analysis helps to understand customer feedback, identify product strengths and weaknesses, and improve the overall customer experience by providing actionable insights.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

### Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Installation

To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/Product-Review-Sentiment-Analysis.git

# Navigate into the project directory
cd Product-Review-Sentiment-Analysis

# Create a virtual environment
python -m venv Product_Review_Analysis

# Activate the virtual environment
# On Windows
Product_Review_Analysis\Scripts\activate
# On macOS/Linux
source Product_Review_Analysis/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

After setting up the project, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
```

This will launch the application in your web browser, where you can input reviews and visualise sentiment analysis results.

### Features

- Sentiment classification of reviews (positive, negative, neutral)
- Visualisation of sentiment distribution across different product categories
- Real-time sentiment prediction for user-input reviews
- Interactive dashboard for exploring customer feedback

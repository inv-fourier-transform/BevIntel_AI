## CodeX BevIntel™ — Price Predictor 🍷

### 📖 Overview

**CodeX BevIntel™** is a web-based price-prediction application built for our client that uses machine learning to estimate appropriate beverage pricing based on a user’s demographic profile, consumption habits, brand familiarity, and packaging & preference choices. The model was trained on a dataset of **40,000+ records** that were originally messy and siloed; extensive cleaning, merging, and preprocessing was done before modeling.

The project includes data cleaning & feature engineering, model training (with LightGBM / XGBoost + scikit-learn), experiment & model versioning (via MLflow + DagsHub), and a user-facing web UI built with Streamlit for interactive price prediction.

---

### 🔧 Tools & Technologies Used

- **Python** — core scripting  
- **pandas** & **NumPy** — for data manipulation, cleaning, numerical operations  
- **scikit-learn** — for preprocessing, encoding, and pipeline utilities  
- **LightGBM** and **XGBoost** — used for model training and experimentation  
- **MLflow** + **DagsHub** — for experiment tracking, model versioning, and dataset/metadata management  
- **Streamlit** — for building the interactive web front-end  
- **seaborn** (and optionally matplotlib) — used during exploratory data analysis and visualizations  

---

### 🧩 Features & Key Concepts

- **Data Cleaning & Integration** — raw data came in silos; manually cleaned, merged, and consolidated into a unified dataset before modelling  
- **Custom feature engineering & business-driven KPIs** — based on client conversations, domain-specific metrics were created that influence pricing predictions, such as:  
  - Zone influence score (reflecting regional / urban-rural/metro impact)  
  - Consumer frequency & brand awareness scores (to capture user’s consumption habits and brand familiarity)  
  - Combined metrics (e.g. zone × income) to reflect intersectional effects  
  - A Brand-Switching Indicator (BSI index) — a flag representing propensity to switch brands based on price/quality/loyalty  
- **Encoding of user preferences & demographics** — transforming categorical user inputs (size preference, health concerns, packaging preference, consumption situation, etc.) into numeric feature vectors via label-encoding and one-hot encoding, matching exactly the structure used in model training  
- **Model persistence and reproducible inference** — trained model saved (with feature metadata and mappings) using `joblib`, enabling future loading and consistent prediction on raw user inputs  
- **Interactive web UI for end-users** — front-end built with Streamlit; users enter their profile and preferences, and get predicted price range instantly  

---

### 🚀 How to Run Locally

Clone the repository, set up environment, and run the app:

```bash
# 1. Clone repo
git clone <your-repo-url>
cd <repo-folder>

# 2. (Optionally) create virtual environment
python -m venv venv
source venv/bin/activate      # On Linux/macOS
# venv\Scripts\activate        # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure the saved model artifact is present:
#    e.g. Artifacts/best_model.joblib

# 5. Run the Streamlit app
streamlit run app.py
```

Then open the URL displayed in your browser (usually `http://localhost:8501`) to access the interface.

---

### 🧠 Usage / Workflow

- User fills in the form (age, gender, consumption habits, preferences) on the Streamlit UI.  
- On clicking **“Calculate Price Range”**, the app gathers inputs into a dictionary.  
- The inputs are passed to a helper function that applies the exact same preprocessing (scoring, feature-engineering, encoding) as done during training.  
- A feature vector is constructed, aligned to the model’s trained feature set.  
- The loaded LightGBM model predicts the price range.  
- The predicted price range is shown to the user on the UI.  

---

### 🎯 Why This Project Matters

Many beverage-pricing models assume a one-size-fits-all approach. But in reality, pricing acceptance depends on a complex mix of demographics, habits, brand familiarity, health concerns, and lifestyle. By combining a large real-world dataset, domain-informed custom KPIs, and machine-learning modeling, **CodeX BevIntel™** offers a data-driven, consumer-segmented approach to price prediction — helping businesses make smarter decisions tailored to different consumer segments.  

---

### 🔒 Data Privacy & Client Confidentiality

> The client data used to build and train the model has **not** been shared. All data cleaning, feature engineering, modelling, and validations were performed internally to ensure confidentiality and compliance with client privacy requirements.  

---

### 📂 Project Structure (Suggested)

```
/
├── app.py              # Streamlit front-end code
├── helper.py           # Preprocessing + feature-engineering helper functions (e.g. prepare_features)
├── Artifacts/
│     └── best_model.joblib   # Saved model + metadata
├── requirements.txt    # All dependencies
└── README.md           # This file
```

---

### 🤝 Contributing & Future Enhancements

- Continue tracking experiments and models using MLflow + DagsHub — e.g. try new models or features, compare performance, maintain version history  
- Improve packaging & deployment — e.g. containerize the app using Docker or deploy on cloud for broader access  
- Add analytics or feedback mechanism in UI to collect real user feedback, enabling future refinements  

---

 

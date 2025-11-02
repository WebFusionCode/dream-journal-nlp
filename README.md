# 💭 Dream Journal NLP Analyzer

A powerful **Streamlit web app** that analyzes your dream journal entries using **Natural Language Processing (NLP)**.  
It uncovers hidden **emotions**, **themes**, **sentiments**, and **topics**, helping you understand your subconscious mind through data-driven insights.

---

## 🚀 Live App
👉 **[Open the App on Streamlit Cloud](https://<your-app-name>.<your-username>.streamlit.app)**  
*(Replace this with your actual Streamlit URL once deployed.)*

---

## 🧠 Features

### ✨ Text Analysis
- Cleans and preprocesses your dream text.
- Extracts **top keywords** and **dominant topics**.
- Performs **sentiment analysis** using transformers.
- Detects **emotional tones** (joy, fear, sadness, anger, etc.).

### 🎨 Visualization
- Beautiful **word clouds** for frequent terms.
- Interactive **bar charts** and **plots** using Plotly and Matplotlib.
- Network graphs to visualize relationships between words and themes.

### 📄 PDF Reports
- Generates a **personalized Dream Analysis Report**.
- Export insights as a **PDF** using `reportlab`.

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | [Streamlit](https://streamlit.io/) |
| NLP | NLTK, spaCy, scikit-learn, transformers |
| Visualization | Matplotlib, Plotly, WordCloud |
| PDF Export | ReportLab |
| Deployment | Streamlit Cloud |
| Language | Python 3.11 |

---

## ⚙️ Project Structure


dream-journal-nlp/
│
├── app/
│ ├── streamlit_app.py # Main app entry point
│ ├── requirements.txt # Python dependencies
│ └── packages.txt # System dependencies for Streamlit Cloud
│
├── src/
│ ├── preprocess.py # Text cleaning and tokenization
│ ├── analyze.py # Sentiment, keywords, topics
│ ├── emotions.py # Emotion classification
│ ├── summary.py # Text summarization
│ └── reporting.py # PDF report generation
│
└── README.md


---

## 🛠️ Installation (Local Setup)

To run locally:

```bash
git clone https://github.com/<your-username>/dream-journal-nlp.git
cd dream-journal-nlp
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py

☁️ Deployment (Streamlit Cloud)

The app is fully compatible with Streamlit Cloud.
To deploy:

Push your latest code to GitHub (main branch).

Go to streamlit.io/cloud
.

Click “New App” → Connect your GitHub repo.

Set the main file path to:
app/streamlit_app.py

Done! Streamlit will automatically install all dependencies and launch your app.


🧩 System Dependencies

Make sure to include a packages.txt file for Streamlit Cloud (to support ReportLab fonts):

libfreetype6-dev
libxft-dev

📚 Acknowledgements

Streamlit

NLTK

spaCy

Hugging Face Transformers

ReportLab

Plotly

🧑‍💻 Author

Harsh Singh (WebFusionCode)
🌐 GitHub

"Explore your subconscious, one dream at a time." 💤

---

Would you like me to personalize this README with your **actual Streamlit app link** (so it’s ready to share)?  
If you share your deployed app’s URL (e.g., `https://dream-journal-nlp.streamlit.app`), I’ll update it and give you the final version.

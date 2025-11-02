# 🌙 Dream Journal NLP App

**Live Demo:** 👉 [https://dreams-psychology.streamlit.app/](https://dreams-psychology.streamlit.app/)

A Streamlit-powered application that helps users analyze their dream journal entries using **Natural Language Processing (NLP)**, **sentiment analysis**, and **topic modeling**.  
It provides deep insights into recurring emotions, themes, and keywords across your dreams — a blend of psychology and AI! 🧠💭

---

## 🚀 Features

- 🧠 **Sentiment Analysis** – Detects positive, neutral, or negative tone in your dreams.  
- ❤️ **Emotion Detection** – Identifies core emotions such as joy, fear, sadness, or anger.  
- 🗝️ **Keyword Extraction** – Finds the most important and recurring dream elements.  
- 🪶 **Topic Modeling** – Clusters dreams into psychological themes using LDA/NMF.  
- ☁️ **Word Clouds** – Visualizes frequently appearing terms.  
- 📊 **Interactive Visuals** – Built with Plotly and Matplotlib for dynamic exploration.  
- 📄 **PDF Report Generation** – Export your dream insights using ReportLab.  
- 🌐 **Deployed on Streamlit Cloud** – No setup needed, just open the app link!

---

## 🧩 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend** | Streamlit |
| **Backend** | Python (Fast computations) |
| **NLP** | SpaCy · NLTK · Transformers |
| **ML/AI** | scikit-learn · sentence-transformers |
| **Visualization** | Plotly · Matplotlib · WordCloud |
| **Exporting** | ReportLab |
| **Hosting** | Streamlit Cloud |

---

## 🧠 Folder Structure

dream-journal-nlp/
│
├── app/
│ └── streamlit_app.py # Main Streamlit entry point
│
├── src/
│ ├── analyze.py # Sentiment, keywords, and topic modeling
│ ├── emotions.py # Emotion classification logic
│ ├── reporting.py # PDF generation with ReportLab
│ └── summary.py # NLP-based summary generation
│
├── requirements.txt # Python dependencies
├── packages.txt # System-level packages for Streamlit Cloud
├── README.md # Project documentation
└── .streamlit/
└── config.toml # Streamlit theme and settings


---

## ⚙️ Local Installation

If you’d like to run the app locally instead of on Streamlit Cloud:

```bash
# 1️⃣ Clone the repo
git clone https://github.com/WebFusionCode/dream-journal-nlp.git
cd dream-journal-nlp

# 2️⃣ Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # (use .venv\Scripts\activate on Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the app
streamlit run app/streamlit_app.py
Then open http://localhost:8501
 in your browser 🌐

☁️ Deployment (for developers)

This app is fully configured for Streamlit Cloud.
It installs both Python and system packages automatically via:

requirements.txt

packages.txt (includes libfreetype6-dev for ReportLab)

To deploy:

Push your changes to GitHub.

Go to streamlit.io/cloud

Click “New App” → connect your repo → select branch and main file:

app/streamlit_app.py


Deploy — done ✅

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a PR or issue if you’d like to collaborate.

🧘 Author 

Developed by WebFusionCode( Harsh Singh )

Dream deeper. Reflect smarter. 💤✨

📜 License

This project is licensed under the MIT License – free to use, modify, and distribute.

⭐ If you like this project, don’t forget to star the repo!
🌙 dreams-psychology.streamlit.app — explore your subconscious through AI.
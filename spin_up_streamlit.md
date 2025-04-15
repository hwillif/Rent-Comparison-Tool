python -m venv venv

windows: 
.\venv\Scripts\activate

mac:
source ./venv/bin/activate

pip install streamlit

streamlit run app.py

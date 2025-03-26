# Autism Spectrum (ASD) Prediction Tool -> Random Forest Classifier
  

A machine learning-powered web application that assesses Autism Spectrum Disorder (ASD) traits across different age groups using Streamlit for UI and Flask for model serving.

  
  Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. **Start Flask Backend** (in separate terminal):
   ```bash
   python flaskapp.py
   ```

2. **Start Streamlit Frontend**:
   ```bash
   python -m streamlit run app.py
   ```

3. Access the app at `http://localhost:8501`

## 🧠 Model Architecture
```bash
models/
├── children_asd_model.pkl
├── adolescent_asd_model.pkl
├── young_asd_model.pkl
└── adult_asd_model.pkl
```
- Models trained using scikit-learn also added scaled pck files too
- StandardScaler used for feature normalization

## 📡 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_questions` | GET | Returns age-appropriate questions |
| `/predict` | POST | Returns prediction (0/1) |

## 🧩 Project Structure
```bash
.
├── s.py                  # Streamlit frontend
├── f.py                  # Flask API server
├── requirements.txt      # Dependencies
└── models/               # models and scalers
└── metrices/             # ROC Curve and classification Report
```

## 🤝 Contributing
1. Fork the project
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit changes:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push to branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact
Joel Siby - [@joelsiby02](https://github.com/joelsiby02)  
Project Link: [https://github.com/joelsiby02/ASD-RandomForest](https://github.com/joelsiby02/Autism-RandomForest)

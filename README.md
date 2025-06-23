# Machine Learning Projects

## 🌸 Iris Species Classification (Jupyter Notebook)
**Predict flower species using measurements**

### Quick Start
1. Install requirements:
```bash
pip install pandas scikit-learn matplotlib notebook
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open and run [`iris-data.ipynb`](iris-data.ipynb)

### Key Features
- Interactive data exploration
- Step-by-step model building
- Visualizations of decision boundaries
- 97% accuracy Decision Tree classifier

---

## 🔢 MNIST Digit Classification (Streamlit App)
**Recognize handwritten digits 0-9**

### Quick Start
```bash
pip install tensorflow==2.12.0 streamlit matplotlib numpy
streamlit run mnist_streamlit.py
```

### Key Features
- CNN model (>95% accuracy)
- Interactive web interface
- Real-time predictions

---

## 📂 Repository Structure
```
.
├── iris-data.ipynb          # Interactive Iris analysis (Jupyter)
├── mnist_streamlit.py       # MNIST web app
├── assets/                  # Sample images
│   ├── iris_visual.png
│   └── mnist_sample.png
└── README.md
```

## 🛠 Installation Notes

### For Iris Notebook:
```bash
pip install jupyter scikit-learn pandas matplotlib
```

### For MNIST App (Python 3.11 required):
```bash
python -m pip install tensorflow==2.12.0 streamlit
```

## 🔍 How to Use

1. **Iris Analysis**:
   - Run cells sequentially in `iris-data.ipynb`
   - Includes data visualization and model evaluation

2. **MNIST Classifier**:
   ```bash
   streamlit run mnist_streamlit.py
   ```
   - Click "Train Model" button
   - View test accuracy and sample predictions

## 📊 Results Summary
| Project | Format | Accuracy | Key Tech |
|---------|--------|----------|----------|
| Iris | Jupyter Notebook | 97% | scikit-learn |
| MNIST | Streamlit App | 95%+ | TensorFlow |


# 🧠 Arabic Sentiment Analysis Framework

A comprehensive, end-to-end Machine Learning and Deep Learning pipeline designed for multidialectal Arabic Sentiment Analysis, with a strong focus on Jordanian dialects (Ammani, Bedouin, Falahi, and Karaki) alongside Modern Standard Arabic (MSA). 

This repository covers everything from heterogeneous data collection and preprocessing, classical statistical baselines, multi-layer perceptrons, and sentence embeddings, up to fine-tuning a state-of-the-art pre-trained Transformer architecture (**MARBERTv2**) and deploying it via an interactive **Streamlit UI**.

---

## 🚀 Key Features
- **Multi-Dialect Dataset Ingestion:** Consolidates, standardizes, and cleans 8 distinct regional Arabic datasets into a single high-quality master corpus.
- **Advanced Text Normalization:** Built-in custom preprocessing pipelines that strip diacritics (*Harakat*) and map complex letters (`أ/إ/آ` $\rightarrow$ `ا`, `ة` $\rightarrow$ `ه`, `ي` $\rightarrow$ `ى`) to handle phonetic variants across dialects.
- **Diverse Model Benchmarking:** Accommodates comparative benchmarking across four distinct methodologies:
  - Linear Classifiers (TF-IDF + Balanced Logistic Regression)
  - Non-Linear Classifiers (TF-IDF + Multi-Layer Perceptron Neural Networks)
  - Dense Dense Semantic Projections (Multilingual MiniLM Embeddings)
  - Deep Learning Fine-Tuning (State-of-the-Art **MARBERTv2** Transformer)
- **Interactive UI Dashboard:** A fully real-time Streamlit web-based UI tailored for native Arabic scripts (RTL text alignment support) equipped with probabilistic metric callouts.

---

## 📊 1. Dataset Consolidation & Cleaning

The framework merges and standardizes multiple dialectal resources located under `./data/` to train models capable of grasping local idioms, stylistic nuances, and standard prose.

### Consolidated Data Streams:
1. **Ammani (Urban):** `Ammani_Zaid.csv`, `urban(ammani) dialect.csv`, `urban-1.csv`, `fullData-Amman.csv`
2. **Bedouin Dialect:** `Bedouin.csv`
3. **Falahi Dialect:** `Falahi Dialect.csv`, `falahia_merged_final.csv`
4. **Karaki Dialect:** `Karaki Sentences V2(1).csv`

### Data Standardization Pipeline:
- **Feature Alignment:** Discards unrelated metadata columns (`id`, `confidence`, `source`, `label_guess`, `Dialect`, `المعنى`) and maps text/classification targets into a uniform structure: `text` and `sentiment`.
- **Target Variable Mapping:** Cleans inconsistent annotations (e.g., mixing `Neutral`, `neutral `, `nuetral`, `Mixed` into a unified `neutral` category).
- **Null Value Resolution:** Systematically filters out structural artifacts and missing tokens to produce a master dataset saved securely at `./cleanedDataset/cleanedDataset.csv`.

---

## 🧪 2. Benchmarking & Model Exploration

The framework splits the unified dataset into an **80/20 train-test split** (using a fixed random state for reproducibility) and applies multiple modeling strategies.

### Preprocessing & Vectorization
For statistical models, a customized pipeline strips out custom Arabic stop words and generates unigrams and bigrams using a `TfidfVectorizer` capped at **5,000 max features**:
- **N-gram Range:** `(1, 2)`
- **Normalization Rules:** Removes diacritics, transforms variations of *Hamza*, maps *Teh Marbuta* to *Heh*, and unifies *Yeh/Alef Maksura*.

### Performance Leaderboard

| Model Architecture | Feature Representation | Optimization Strategy / Parameters | Evaluation Accuracy |
| :--- | :--- | :--- | :---: |
| **MARBERTv2 Transformer** | Wordpiece Tokenization | Fine-tuned 3 Epochs, AdamW, $\text{LR}=2e-5$ | **78.43%** |
| **Logistic Regression** | TF-IDF (Unigrams/Bigrams) | Inverse Regularization Strength $C=2.0$, Balanced weights | **67.40%** |
| **Multi-Layer Perceptron (MLP)** | TF-IDF (Unigrams/Bigrams) | Dense Hidden Topology: `(128, 64)`, Adam Optimizer | **62.30%** |
| **Sentence-Transformer + LogReg** | `paraphrase-multilingual-MiniLM-L12-v2` | Dense Embeddings + Standard Scaler pipeline | **57.55%** |

---

## 👑 3. Deep Learning Fine-Tuning: MARBERTv2

The crown jewel of this framework is the contextualized fine-tuning of `UBC-NLP/MARBERTv2`, a massive language model specifically optimized for dialectal Arabic text representation.

### Training Configuration
Using the Hugging Face `Trainer` API, the model is fine-tuned under the following structural configuration:
```python
training_args = TrainingArguments(
    output_dir="./arabic_dialect_transformer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)
```

### Training Convergence Profile
During the training process, the model demonstrates rapid convergence with stable validation accuracy:
- **Epoch 1:** Validation Loss: `0.5457`, Evaluation Accuracy: `78.43%`
- **Epoch 2:** Validation Loss: `0.5679`, Evaluation Accuracy: `78.57%`
- **Epoch 3:** Validation Loss: `0.6301`, Evaluation Accuracy: `78.50%`

The optimal weight snapshot is captured at **Epoch 1** (lowest validation loss to prevent overfitting) and saved natively inside directory `./my_best_arabic_sentiment_model`.

---

## 💻 4. Interactive Streamlit Interface

The project comes with a clean, user-friendly **Streamlit GUI application** that loads the deep learning model pipeline and lets users run interactive sentiment inference on real-time text input.

### Front-End & Engine Highlights:
- **Asynchronous Execution Fixes:** Integrates platform-specific `asyncio` policy handling for robust execution across Windows and Linux environments.
- **Resource Caching:** Wraps model loading inside `@st.cache_resource` to guarantee instantaneous webapp initialization and eliminate memory leakages on button clicks.
- **Native Alignment:** Inject explicit CSS styles into the engine to force native Right-to-Left (RTL) formatting on textual inputs, ensuring a seamless experience for Arabic speakers.
- **Confidence Metrics:** Returns descriptive visual outputs classified dynamically across three confidence spectrums:
  - 😡 **سلبي (Negative)**
  - 😐 **محايد (Neutral)**
  - 😍 **إيجابي (Positive)**

---

## 📁 5. Project Repository Structure

```directory
├── data/                                 # Raw multi-dialect CSV files
│   ├── Ammani_Zaid.csv
│   ├── Bedouin.csv
│   ├── Falahi Dialect.csv
│   ├── falahia_merged_final.csv
│   ├── fullData-Amman.csv
│   ├── Karaki Sentences V2(1).csv
│   ├── urban(ammani) dialect.csv
│   └── urban-1.csv
├── cleanedDataset/
│   └── cleanedDataset.csv                # Merged, standardized dataset
├── my_best_arabic_sentiment_model/       # Saved MARBERTv2 fine-tuned artifacts
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── data_cleaning.ipynb                   # Initial dataset consolidation notebook
├── model_benchmarking.ipynb              # Baseline evaluations and MARBERTv2 training
├── app.py                                # Production Streamlit UI Application script
└── README.md                             # Repository documentation
```

---

## ⚙️ 6. Setup and Installation

Follow these steps to run the dataset cleaning, model benchmarking, or the interactive application on your local machine:

### 1. Clone & Set Up the Environment
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Project Dependencies
Ensure you have the required core packages installed:
```bash
pip install torch transformers datasets streamlit pandas numpy scikit-learn sentence-transformers evaluate huggingface_hub openpyxl
```

### 3. Run the Streamlit GUI Application
Launch the real-time user interface from the root directory:
```bash
streamlit run app.py
```

---

## 📝 Usage Example (Inference)

You can run sentiment analysis programmatically using the following snippet:

```python
from transformers import pipeline

# Load fine-tuned pipeline
pipe = pipeline("text-classification", model="./my_best_arabic_sentiment_model")

# Sample dialectal review (Jordanian Ammani/Urban)
text = "الخدمة كانت ممتازة جداً والتوصيل سريع"
result = pipe(text)[0]

print(f"Predicted Class: {result['label']} with confidence {result['score']*100:.2f}%")
```

---
*Developed as a comprehensive framework for Arabic Dialectal Pattern Recognition and Natural Language Processing.*

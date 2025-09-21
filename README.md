# 📧 Phishing Email Classifier: A Dual-Model Defense System 🛡️

Welcome to a fascinating journey into the world of email security, where we apply the power of Natural Language Processing (NLP) to combat the pervasive threat of phishing. This project presents a unique dual-model approach, pitting a classic machine learning model against a state-of-the-art deep learning architecture to unmask digital deception.

## ✨ Project Grand Vision ✨

In our interconnected digital landscape, phishing attacks lurk as ever-present dangers. They evolve, they adapt, and they strive to trick us. This project rises to the challenge by building two distinct, yet equally impressive, classification models:

* **Model-1: Naive Bayes (The Classic Detective 🔍)**: This notebook implements a probabilistic classifier that serves as a strong and computationally efficient baseline. It's a testament to the fact that sometimes, the simplest solutions can be the most effective.
* **Model-2: BERT (The AI Linguist 🧠)**: Here, we embrace the cutting edge of deep learning. By fine-tuning a pre-trained Bidirectional Encoder Representations from Transformers (BERT) model, we've created a classifier that understands the nuanced context and semantics of language, achieving a level of performance that was once unimaginable.

Our core goal is to compare the performance, complexity, and resource requirements of these two vastly different modeling paradigms.

---

## 📈 Model Performance & Key Insights

Both models were rigorously evaluated on the same held-out test set to ensure a fair comparison. The results are nothing short of impressive:

| Feature | Naive Bayes Model | BERT Model |
| :--- | :--- | :--- |
| **Accuracy** | 99.33% | 98.98% |
| **Precision** | 0.99 | 0.99 |
| **Recall** | 0.99 | 0.99 |
| **F1-Score** | 0.99 | 0.99 |

<br>

**Deciphering the Results:**

* **The Race for Accuracy:** While the **Naive Bayes** model technically achieved a slightly higher overall accuracy, its performance is based on keyword frequency. It's incredibly fast and lightweight.
* **The Power of Context:** The **BERT** model, despite a minuscule difference in accuracy, excels at understanding the contextual and syntactic features of an email. This means it can identify sophisticated phishing attempts that don't rely on obvious keywords. This superior contextual understanding comes at the cost of significantly longer training times and a higher demand for computational resources (a GPU is highly recommended).

Ultimately, your choice depends on your needs: speed and simplicity, or a deeper, more robust understanding of language.

---

## 📂 Project Structure

* `Model-1(Naive Bayes).ipynb`: The notebook containing our classic Naive Bayes classifier.
* `Model-2(BERT).ipynb`: The notebook showcasing the powerful BERT-based classifier.
* `phishing_email.csv`: The heart of the project—the dataset used for training and evaluation.

---

## 🚀 Getting Started

Ready to see these models in action? Follow these simple steps to run the notebooks on your local machine.

### Prerequisites

* **Python 3.x**
* **Jupyter Notebook** or **JupyterLab**

### Installation

1.  Clone this repository to your local machine.
    ```bash
    git clone [https://github.com/shriya-pargi/Phishing-Email-Classifier.git](https://github.com/shriya-pargi/Phishing-Email-Classifier.git)
    ```
2.  Navigate to the project directory.
    ```bash
    cd Phishing-Email-Classifier
    ```
3.  Install the required libraries. For Naive Bayes, you'll need `pandas`, `scikit-learn`, and `matplotlib`. For BERT, you'll also need `torch` and `transformers`.
    ```bash
    pip install pandas scikit-learn matplotlib torch transformers
    ```

### How to Run

1.  Open a terminal in the project directory and launch Jupyter.
    ```bash
    jupyter notebook
    ```
2.  In the Jupyter interface, open either notebook and run the cells sequentially to reproduce the results.

---

## 🙏 Acknowledgements

This project was a rewarding challenge and a fantastic learning experience. Special thanks to the creators of the `scikit-learn` and `Hugging Face Transformers` libraries for providing the incredible tools that make this work possible.

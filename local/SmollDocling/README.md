
---

## 📌 Notebooks Overview

### 🔧 `DataPreprocess.ipynb`
- Converts original annotated prescription data to the **DocTag** format.

### 📦 `DataLoader.ipynb`
- Wraps the preprocessed dataset using the **Hugging Face `datasets`** library.
- Enables efficient integration into the training pipeline.
- Includes tokenization and splitting into train/validation sets.

### 🎯 `FT.ipynb`
- Fine-tunes the model using **LoRA (Low-Rank Adaptation)** for efficient training.
- Supports multiple training configurations for different experimental setups.

### 📊 `Eval.ipynb`
- Evaluates the fine-tuned model on a validation or test set.
- Outputs precision, recall, F1 score, and optionally a detailed error analysis.

### 📄 `result.json`
- Stores evaluation outputs such as predicted labels, confidence scores, and associated metadata.

---

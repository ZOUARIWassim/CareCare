# 🧾 OCR-Based Medical Prescription Information Extraction

This project implements an end-to-end pipeline for extracting structured information from scanned French medical prescriptions using a combination of PaddleOCR and a fine-tuned TinyBERT model. The extracted fields include doctor names, identification numbers (RPPS, FINESS), prescription content, and dates.

## 🔧 Project Structure

```
.
├── preparation_fine_tuning_dataset.py     # Prepares training CSV from annotated JSON
├── tinyBert_finetuning.py                 # Fine-tunes TinyBERT on labeled OCR entities
├── test_tinyBert.py                       # Evaluates TinyBERT on a test dataset
├── PaddleOCR_tinyBert.py                  # Performs OCR and classification using TinyBERT
├── finetuned_tinyBert_implementation.py   # End-to-end script: image -> OCR -> label -> CSV
├── test_paddleOCR_Naive_match.py          # Heuristic matching without ML (keyword + regex)
├── to_onnx.py                              # Exports TinyBERT model to ONNX format
```

## 📌 Key Components

### 1. **OCR Engine**
- **Library**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **Function**: Detects and extracts text from prescription images.
- **Language support**: French (`lang="fr"`)

### 2. **Text Classification Model**
- **Backbone**: [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny)
- **Fine-Tuned**: On custom dataset extracted from labeled JSON medical documents.
- **Task**: Classifies OCR-extracted lines into entity types.

### 3. **Regex Post-processing**
- Used for validating or overriding model predictions, especially for fields with strong patterns (e.g., dates, RPPS numbers).

## 🧪 Supported Fields

| Field                   | Example                     | Type         |
|------------------------|-----------------------------|--------------|
| `Nom-du-medecin`       | "Dr Jean Dupont"            | Text         |
| `Numero-RPPS`          | "10101010101"               | Numeric ID   |
| `Numero-AM-Finess`     | "01010101010"               | Numeric ID   |
| `Date-de-la-prescription` | "15/06/2025"             | Date         |
| `Texte-soin-ALD`       | ALD-related prescription     | Text block   |
| `Texte-soin-sans-ALD`  | Other prescriptions          | Text block   |
| `Adresse-prescripteur` | French-style addresses       | Text         |

## 🚀 How to Run

### 1. **Prepare the Dataset**
```bash
python preparation_fine_tuning_dataset.py
```

### 2. **Fine-Tune TinyBERT**
```bash
python tinyBert_finetuning.py
```

### 3. **Evaluate the Fine-Tuned Model**
```bash
python test_tinyBert.py
```

### 4. **Run Full Pipeline on JSON + Base64 Image Files**
```bash
python finetuned_tinyBert_implementation.py
```

### 5. **Export to ONNX (optional for deployment)**
```bash
python to_onnx.py
```

## 🧠 Model Export

Use `to_onnx.py` to convert the fine-tuned TinyBERT model to ONNX format for optimized inference in mobile or embedded environments.

## 📁 Input Format

- Input: `.json` files containing base64-encoded prescription images and labeled entities (like Google Document AI format).
- Output: Structured `.csv` files containing both predictions and ground truth.

## 💡 Alternatives & Tests

- `test_paddleOCR_Naive_match.py`: Rule-based extractor (no ML) for simple use cases.
- `PaddleOCR_tinyBert.py`: Lighter version for quick demo or single image testing.

## ✅ Dependencies

```bash
pip install paddleocr paddlenlp transformers datasets pandas scikit-learn torch optimum
```

## 📌 Notes

- Model trained and tested primarily on French prescriptions.
- For mobile deployment, ONNX export is recommended.
- Regex rules are crafted for French formats (addresses, dates, etc.).

## 👩‍⚕️ Example Output

```
doctor_name: Dr Jean Dupont
rpps_number: 10101010101
finess_number: 01010101010
date: 15/06/2025
prescription_content: ["Pansement ALD", "Ibuprofène 400mg"]
```

## 📬 Contact

For questions, improvements, or deployment help, feel free to reach out.


---

## 📘 `documentAI.ipynb`

This notebook is responsible for:
- Loading prescription documents (PDF or image format)
- Sending requests to **Google Document AI** using the client API
- Parsing the returned structured data (entities, bounding boxes, values)
- Comparing predictions with ground truth labels
- Computing evaluation metrics: accuracy, precision, recall, F1 score

> Make sure you run the backend of the webapp before running this notebook

---

## 📄 `result.json`

- Stores the structured predictions and metrics obtained from the evaluation.

---
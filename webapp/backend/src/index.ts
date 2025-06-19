import express from "express";
import { Router } from "express";
import ocrRouter from "./routes/ocr";
import signatureDetectionRouter from "./routes/signature_detection";
import cors from "cors";

const app = express();
app.use(express.json());

app.use(cors());

app.use('/ocr', ocrRouter);

app.use('/signature-detection', signatureDetectionRouter);

app.listen(3000, () => {
    console.log("Server is running on port 3000");
});

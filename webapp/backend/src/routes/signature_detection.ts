import express, { Request, Response } from "express";
import multer from "multer";
import dotenv from 'dotenv';
dotenv.config();
import { connectApi } from "../utils/API_utils";

const upload = multer({ storage: multer.memoryStorage() });

const router = express.Router();

router.post("/", upload.single("file"), async (req: Request, res: Response): Promise<void> => {
    try {
        if (!req.file) {
            res.status(400).json({ error: "No file uploaded" });
            return;
        }

        console.log("Processing document...");
        const processorId = process.env.PROCESSOR_ID_SIGN;
        if (!processorId) {
            throw new Error("Missing required environment variable: PROCESSOR_ID_EXTRACTION.");
        }
        const { client, name } = connectApi(processorId);
        const request = {
            name,
            rawDocument: {
                content: req.file.buffer.toString("base64"),
                mimeType: req.file.mimetype,
            },
        };

        const [response] = await client.processDocument(request);
        if (!response.document) {
            throw new Error("Failed to process document.");
        }
        res.json(response.document.entities);
        console.log("Document processed successfully");
    } catch (error) {
        console.error("Error processing document:", error);
        res.status(500).json({ error: "Failed to process document" });
    }
}
);

export default router;
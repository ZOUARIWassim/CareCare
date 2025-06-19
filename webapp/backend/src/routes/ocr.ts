import express, { Request, Response } from "express";
import multer from "multer";
import dotenv from "dotenv";
dotenv.config();
import { connectApi } from "../utils/API_utils";
import { fuzzyMatch } from "../utils/fuzzymatch";
import { validateDate } from "../utils/validateDate";


const upload = multer({ storage: multer.memoryStorage() });

const router = express.Router();

router.post("/", upload.single("file"), async (req: Request, res: Response) => {
    try {
        if (!req.file) {
            res.status(400).json({ error: "No file uploaded" });
            return;
        }

        console.log("Processing document...");

        const processorId = process.env.PROCESSOR_ID_EXTRACTION;
        if (!processorId) {
            throw new Error("Missing required environment variable: PROCESSOR_ID_EXTRACTION.");
        }

        const { client, name: processorName } = connectApi(processorId);
        const request = {
            name: processorName,
            rawDocument: {
                content: req.file.buffer.toString("base64"),
                mimeType: req.file.mimetype,
            },
        };

        const [response] = await client.processDocument(request);
        if (!response.document) {
            throw new Error("Failed to process document.");
        }

        const entities = response.document.entities || [];
        console.log(entities)
        // Recherche des entités
        const rppsEntity = entities.find(e => e.type?.toLowerCase().includes("rpps"));
        const nameEntity = entities.find(e => e.type?.toLowerCase().includes("nom"));

        // Variables pour stocker les valeurs extraites
        let rpps: string | null = null;
        let extractedName: string | null = null;

        // Extraction des valeurs si les entités existent
        if (rppsEntity) rpps = rppsEntity.mentionText || null;
        if (nameEntity) extractedName = nameEntity.mentionText || null;

        console.log(`RPPS extrait: ${rpps || "non trouvé"}`);
        console.log(`Nom extrait: ${extractedName || "non trouvé"}`);

        // On tente le matching si au moins une des deux valeurs est présente
        if (rpps || extractedName) {
            console.log("Tentative de matching...");
            const match = await fuzzyMatch(rpps || "none", extractedName || "none");

            if (match) {
                let isRppsAdded=false
                let isNameAdded=false

                console.log("Match found:", match);
                for (const entity of entities) {
                    const type = entity.type?.toLowerCase();
                    
                    if (type === 'numero-rpps' && match.numero_rpps) {
                        entity.mentionText = match.numero_rpps;
                        if (entity.normalizedValue) {
                            entity.normalizedValue.text = match.numero_rpps;
                        }
                        isRppsAdded=true
                    }

                    if (type === 'nom-du-medecin' && match.nom && match.prenom) {
                        entity.mentionText = `${match.prenom} ${match.nom}`;
                        isNameAdded=true
                    }
                }
                if (!isRppsAdded) {
                    entities.push({
                        type: 'numero-rpps',
                        mentionText: match.numero_rpps,
                        normalizedValue: { text: match.numero_rpps }
                    });
                }
                if (!isNameAdded) {
                    entities.push({
                        type: 'nom-du-medecin',
                        mentionText: `${match.prenom} ${match.nom}`,
                        normalizedValue: { text: `${match.prenom} ${match.nom}` }
                    });
                }
            }
        } else {
            console.warn("Aucun champ RPPS ou nom trouvé");
        }

        // Validation de la date
        const dateEntity = entities.find(e => e.type?.toLowerCase().includes("date-de-la-prescription"));
        if (dateEntity) {
            const validDate = validateDate(dateEntity.mentionText || "");
            if (validDate) {
                dateEntity.mentionText = validDate;
                if (dateEntity.normalizedValue) {
                    dateEntity.normalizedValue.text = validDate;
                }
            } else {
                entities.splice(entities.indexOf(dateEntity), 1);
            }
        }

        res.json(entities);

    } catch (error) {
        console.error("Error processing document:", error);
        res.status(500).json({ error: "Failed to process document" });
    }
});

export default router;
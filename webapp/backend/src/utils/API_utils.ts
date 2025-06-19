import { DocumentProcessorServiceClient } from "@google-cloud/documentai";
import { ClientOptions } from "google-gax";


function connectApi(PROCESSOR_ID: string): { client: DocumentProcessorServiceClient; name: string } {
    try {
        const location = process.env.LOCATION;
        const projectId = process.env.PROJECT_ID;
        const processorId = PROCESSOR_ID;

        if (!location || !projectId || !processorId) {
            throw new Error("Missing required environment variables: LOCATION, PROJECT_ID, or PROCESSOR_ID.");
        }

        const clientOptions: ClientOptions = { apiEndpoint: `${location}-documentai.googleapis.com` };
        const client = new DocumentProcessorServiceClient(clientOptions);
        const name = client.processorPath(projectId, location, processorId);

        return { client, name };
    } catch (error) {
        console.error("Error connecting to Google Document AI:", error);
        throw new Error("Failed to connect to Google Document AI.");
    }
}

export { connectApi };
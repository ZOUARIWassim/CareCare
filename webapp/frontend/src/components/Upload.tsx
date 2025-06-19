import "../styles/components/Upload.scss";
import {  useState } from "react";
import CircularProgress from '@mui/material/CircularProgress'; 

const Upload = () => {
    const [data, setData] = useState([]);
    const [docSign, setDocSign] = useState([]);
    const [loading, setLoading] = useState(false);

    function get_doc_class(docSign: any) {
        if (docSign[0].confidence >= docSign[1].confidence) {
            return "Signed";
        } else {
            return "Unsigned";
        }
    }

    async function processDocument(file: File) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:3000/ocr/', {
                method: 'POST',
                body: formData, 
            });

            if (!response.ok) {
                throw new Error('Failed to process the document');
            }

            const data = await response.json();
            setData(data);
        } catch (error) {
            console.error('Error processing document:', error);
        }
    }

    async function classifyDocSign(file: File) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:3000/signature-detection', {
                method: 'POST',
                body: formData, 
            });

            if (!response.ok) {
                throw new Error('Failed to process the document');
            }

            const docSign = await response.json();
            setDocSign(docSign);
        } catch (error) {
            console.error('Error processing document:', error);
        }
    }

    const handleFileUpload = async (event: any) => {
        const file = event.target.files[0];
        if (!file) return;

        setLoading(true);
        try {
            await Promise.all([
                processDocument(file),
                classifyDocSign(file)
            ]);
        } catch (error) {
            console.error('Error processing document:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
        <div className="UploadContainer">
            <label htmlFor="fileUpload" className="UploadLabel">Upload Your Prescription</label>
            <input id="fileUpload" type="file" accept="image/*" className="UploadInput" onChange={handleFileUpload} />
        </div>
        <div className="ExtractionContainer">
            <h1>Extraction</h1>
            <div className="Extraction">
                {loading ? (
                        <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
                            <CircularProgress /> 
                        </div>
                    ) : (
                        <>
                            <ul>
                                {data.map((entity: any, index: number) => (
                                    <li key={index}>
                                        <strong>{entity.type}</strong> - {entity.mentionText}
                                    </li>
                                ))}
                                {docSign.length > 0 && (
                                    <li>
                                        <strong>Signature</strong> - {get_doc_class(docSign)}
                                    </li>
                                )}
                            </ul>

                        </>
                    )}
            </div>
        </div>
        </>
    )
}

export default Upload
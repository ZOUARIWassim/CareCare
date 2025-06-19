import { Client } from 'pg';
import fuzz from 'fuzzball';
import dotenv from 'dotenv';

dotenv.config();

async function getRowsFromDB(): Promise<any[]> {
    const client = new Client({
        host: process.env.DB_HOST,
        port: parseInt(process.env.DB_PORT || '5432'),
        user: process.env.DB_USER,
        password: process.env.DB_PASSWORD,
        database: process.env.DB_NAME,
    });

    await client.connect();
    const res = await client.query('SELECT * FROM Personne_activite');
    await client.end();
    return res.rows;
}

export async function fuzzyMatch(rppsNumber: string | null, drName: string | null): Promise<any | null> {
    const rows = await getRowsFromDB();
    let bestValue = -Infinity;
    let bestRow: any | null = null;

    const drNameLower = drName?.toLowerCase().trim() || '';
    const rppsInput = rppsNumber?.trim() || '';

    for (const row of rows) {
        let scoreRpps = 0;
        let scoreName = 0;
        let totalScore = 0;

        const rpps = (row.numero_rpps || '').trim();
        const fullName = `${row.nom || ''} ${row.prenom || ''}`.toLowerCase().trim();

        if (rppsInput) {
            scoreRpps = fuzz.ratio(rppsInput, rpps);
        }

        if (drNameLower) {
            scoreName = fuzz.token_sort_ratio(drNameLower, fullName);
        }

        if (rppsInput && drNameLower) {
            totalScore = scoreRpps * 0.6 + scoreName * 0.4;
        } else if (rppsInput) {
            totalScore = scoreRpps;
        } else if (drNameLower) {
            totalScore = scoreName;
        } else {
            continue; // Aucun critère, on skip
        }

        if (totalScore > 90) {
            console.log(`✅ Strong match: RPPS=${rpps}, Name=${row.nom} ${row.prenom}`);
            return row;
        }

        if (totalScore > bestValue && totalScore>50) {
            bestValue = totalScore;
            bestRow = row;
        }
    }

    if (bestRow) {
        console.log(`🤔 Best fuzzy match: ${bestRow.numero_rpps} ${bestRow.nom} ${bestRow.prenom} with score: ${bestValue.toFixed(2)}`);
    } else {
        console.log("❌ No match found.");
    }

    return bestRow;
}

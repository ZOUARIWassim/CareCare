import { parse, isAfter, format, isValid } from 'date-fns';
import { fr } from 'date-fns/locale';


export function validateDate(dateStr: string): string | null {
    const formats = [
        "dd/MM/yyyy",
        "dd MMMM yyyy",
        "dd MMM yyyy"
    ];

    for (const fmt of formats) {
        const parsed = parse(dateStr, fmt, new Date(), { locale: fr });

        if (isValid(parsed) && !isAfter(parsed, new Date())) {
            return format(parsed, "dd/MM/yyyy");
        }
    }

    return null;
}

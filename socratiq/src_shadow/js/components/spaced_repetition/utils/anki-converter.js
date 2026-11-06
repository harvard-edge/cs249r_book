import { MAX_IMPORT_BYTES, validateFlashcardImport } from './import-validation.js';

export class AnkiConverter {
    static parseAnkiCSV(csvContent) {
        try {
            if (typeof csvContent !== 'string' || csvContent.length > MAX_IMPORT_BYTES) throw new Error('Import file is too large');
            const lines = csvContent.replace(/^\uFEFF/, '').split(/\r?\n/);
            const headers = this.parseHeaders(lines);
            if (!headers.separator) {
                const firstRow = lines[headers.headerEndIndex] || '';
                headers.separator = firstRow.includes('\t') ? '\t' : firstRow.includes(';') ? ';' : ',';
            }
            if (headers.separator.length !== 1 || /["\r\n]/.test(headers.separator)) throw new Error('Invalid separator');
            const rows = this.parseRows(lines.slice(headers.headerEndIndex).join('\n'), headers.separator);
            if (rows.length && /^(question|front)$/i.test(rows[0][0].trim()) && /^(answer|back)$/i.test(rows[0][1]?.trim())) rows.shift();
            if (!rows.length || rows.length > 10000) throw new Error('No cards or too many cards');
            const cards = rows.map(fields => {
                if (fields.length < 2 || fields.length > 3) throw new Error('Invalid card columns');
                return { question: fields[0], answer: fields[1], tags: fields[2] ? fields[2].split(/\s+/).filter(Boolean) : [] };
            });
            return validateFlashcardImport(this.convertToAppFormat(cards, headers));
        } catch (error) {
            console.error('Error parsing Anki CSV:', error);
            throw new Error('Invalid Anki file format');
        }
    }
    
    static parseHeaders(lines) {
        const headers = {
            separator: null,
            html: false,
            tags: [],
            notetype: null,
            deck: null,
            headerEndIndex: 0
        };
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) { headers.headerEndIndex = i + 1; continue; }
            if (line.startsWith('#')) {
                const colon = line.indexOf(':');
                const key = line.slice(1, colon < 0 ? undefined : colon).trim();
                const value = colon < 0 ? '' : line.slice(colon + 1).trim();
                switch (key) {
                    case 'separator':
                        headers.separator = this.getSeparator(value);
                        break;
                    case 'html':
                        headers.html = value === 'true';
                        break;
                    case 'tags':
                        headers.tags = value.split(' ').filter(Boolean);
                        break;
                    case 'notetype':
                        headers.notetype = value;
                        break;
                    case 'deck':
                        headers.deck = value;
                        break;
                }
                headers.headerEndIndex = i + 1;
            } else {
                break;
            }
        }
        
        return headers;
    }
    
    static getSeparator(value) {
        const separators = {
            'comma': ',',
            'semicolon': ';',
            'tab': '\t',
            'space': ' ',
            'pipe': '|',
            'colon': ':'
        };
        return separators[value.toLowerCase()] || value;
    }
    
    static parseRows(content, separator) {
        const rows = [];
        let row = [];
        let field = '';
        let quoted = false;
        let closed = false;
        const finishRow = () => {
            row.push(field);
            if (row.some(value => value.trim())) rows.push(row);
            row = [];
            field = '';
            closed = false;
        };
        for (let index = 0; index < content.length; index++) {
            const char = content[index];
            if (!quoted && row.length === 0 && !field.trim() && char === '#') {
                while (index < content.length && content[index] !== '\n') index++;
                field = '';
                continue;
            }
            if (quoted) {
                if (char === '"') {
                    if (content[index + 1] === '"') { field += '"'; index++; }
                    else { quoted = false; closed = true; }
                } else field += char;
            } else if (char === separator) {
                row.push(field); field = ''; closed = false;
            } else if (char === '\n') {
                finishRow();
            } else if (char === '"' && field === '' && !closed) {
                quoted = true;
            } else if (closed) {
                if (!/\s/.test(char)) throw new Error('Unexpected text after quoted field');
            } else {
                field += char;
            }
        }
        if (quoted) throw new Error('Unclosed quoted field');
        finishRow();
        return rows;
    }

    static convertToAppFormat(cards, headers) {
        const chapter = {
            chapter: 0,
            title: headers.deck || "Imported Deck"
        };
        
        const processedCards = cards.map(card => ({
            id: crypto.randomUUID(),
            chapter: chapter.chapter,
            question: this.processAnkiField(card.question, headers.html),
            answer: this.processAnkiField(card.answer, headers.html),
            created: new Date().toISOString(),
            repetitions: 0,
            easeFactor: 2.5,
            interval: 0,
            nextReviewDate: null,
            lastReviewQuality: 0,
            tags: [...new Set([...card.tags, ...headers.tags])]
        }));
        
        return {
            chapter_card_sets: [{
                chapter: 0,
                cards: processedCards
            }],
            current_chapter: chapter
        };
    }
    
    static processAnkiField(field, allowHtml) {
        if (!allowHtml) {
            // Escape HTML characters
            field = field.replace(/&/g, '&amp;')
                       .replace(/</g, '&lt;')
                       .replace(/>/g, '&gt;');
        }
        
        // Convert Anki media references
        field = field.replace(/\[sound:(.*?)\]/g, (match, filename) => 
            `<audio controls src="${encodeURIComponent(filename)}"></audio>`);
            
        return field;
    }
    
    static convertToAnkiFormat(appData) {
        const lines = [];
        
        // Add headers
        lines.push('#separator:Semicolon');
        lines.push('#html:true');
        if (appData.current_chapter?.title) {
            lines.push(`#deck:${appData.current_chapter.title}`);
        }
        
        // Add column headers
        lines.push('Question;Answer;Tags');
        
        // Add cards
        for (const chapterSet of appData.chapter_card_sets) {
            for (const card of chapterSet.cards) {
                const question = this.escapeField(card.question);
                const answer = this.escapeField(card.answer);
                const tags = this.escapeField((card.tags || []).join(' '));
                
                lines.push(`${question};${answer};${tags}`);
            }
        }
        
        return lines.join('\n');
    }
    
    static escapeField(field) {
        if (field.includes(';') || field.includes('\n') || field.includes('"')) {
            // Escape quotes and wrap in quotes
            return `"${field.replace(/"/g, '""')}"`;
        }
        return field;
    }
} 
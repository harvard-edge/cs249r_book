export class AnkiConverter {
    static parseAnkiCSV(csvContent) {
        try {
            // Split content into lines and filter out empty lines
            const lines = csvContent.split('\n').filter(line => line.trim());
            
            // Parse headers
            const headers = this.parseHeaders(lines);
            const startIndex = headers.headerEndIndex;
            const separator = headers.separator || ';';
            
            // Parse the column definitions line
            const columnLine = lines[startIndex];
            const columns = this.parseColumns(columnLine, separator);
            
            // Parse the actual data
            const cards = [];
            for (let i = startIndex + 1; i < lines.length; i++) {
                const line = lines[i];
                if (line.startsWith('#')) continue; // Skip comments
                
                const card = this.parseCardLine(line, separator, columns);
                if (card) cards.push(card);
            }
            
            // Convert to our app's format
            return this.convertToAppFormat(cards, headers);
        } catch (error) {
            console.error('Error parsing Anki CSV:', error);
            throw new Error('Invalid Anki file format');
        }
    }
    
    static parseHeaders(lines) {
        const headers = {
            separator: ';',
            html: false,
            tags: [],
            notetype: null,
            deck: null,
            headerEndIndex: 0
        };
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('#')) {
                const [key, value] = line.substring(1).split(':').map(s => s.trim());
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
            'Comma': ',',
            'Semicolon': ';',
            'Tab': '\t',
            'Space': ' ',
            'Pipe': '|',
            'Colon': ':'
        };
        return separators[value] || value;
    }
    
    static parseColumns(line, separator) {
        return line.split(separator).map(col => col.trim());
    }
    
    static parseCardLine(line, separator, columns) {
        // Handle quoted fields with embedded separators
        const fields = [];
        let currentField = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                if (inQuotes && line[i + 1] === '"') {
                    // Handle escaped quotes
                    currentField += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === separator && !inQuotes) {
                fields.push(currentField);
                currentField = '';
            } else {
                currentField += char;
            }
        }
        fields.push(currentField);
        
        if (fields.length < 2) return null;
        
        return {
            question: fields[0],
            answer: fields[1],
            tags: fields[2] ? fields[2].split(' ').filter(Boolean) : []
        };
    }
    
    static convertToAppFormat(cards, headers) {
        const chapter = {
            chapter: 0,
            title: headers.deck || "Imported Deck"
        };
        
        const processedCards = cards.map(card => ({
            id: crypto.randomUUID(),
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
            `<audio controls src="${filename}"></audio>`);
            
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
                const tags = card.tags.join(' ');
                
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
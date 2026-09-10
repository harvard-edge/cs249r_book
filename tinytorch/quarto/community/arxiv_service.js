export class ArxivService {
    constructor() {
        this.category = "cat:cs.AI";
        this.maxResults = 10;
        this.papers = [];
        this.fetching = false;
        this.nextStart = 0;
    }

    async fetchPapers() {
        if (this.fetching) return;
        this.fetching = true;

        try {
            // Using custom Supabase Edge Function to bypass CORS and manage rate limits
            const url = new URL("https://zrvmjrxhokwwmjacyhpq.supabase.co/functions/v1/arxiv-get");
            url.searchParams.append('search_query', this.category);
            url.searchParams.append('sortBy', 'submittedDate');
            url.searchParams.append('sortOrder', 'descending');
            url.searchParams.append('start', this.nextStart);
            url.searchParams.append('max_results', this.maxResults);

            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'apikey': 'sb_publishable_AP2UzNWC3T1GQGjtuTr_PQ_9q6l7AC0'
                }
            });
            const str = await response.text();
            
            // Parse XML
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(str, "text/xml");
            const entries = xmlDoc.getElementsByTagName("entry");
            
            for (let i = 0; i < entries.length; i++) {
                const entry = entries[i];
                const titleNode = entry.getElementsByTagName("title")[0];
                const publishedNode = entry.getElementsByTagName("published")[0];
                
                if (!titleNode || !publishedNode) continue;

                const title = titleNode.textContent.replace(/\n/g, ' ').trim();
                
                // Authors
                const authorNodes = Array.from(entry.getElementsByTagName("author"));
                let authors = "Unknown";
                if (authorNodes.length > 0) {
                    const names = authorNodes.map(a => {
                        const n = a.getElementsByTagName("name")[0];
                        return n ? n.textContent : "";
                    }).filter(n => n);
                    
                    if (names.length > 2) {
                        authors = names.slice(0, 2).join(', ') + " et al.";
                    } else {
                        authors = names.join(', ');
                    }
                }

                const published = publishedNode.textContent;
                const year = new Date(published).getFullYear();
                
                this.papers.push({
                    year: year,
                    name: title,
                    authors: authors,
                    isPaper: true,
                    // Use a unique ID to avoid dupes if api acts up?
                    id: entry.getElementsByTagName("id")[0]?.textContent
                });
            }
            
            this.nextStart += this.maxResults;
            
        } catch (e) {
            console.error("Arxiv fetch failed", e);
        } finally {
            this.fetching = false;
        }
    }

    async getNextPaper() {
        // Buffer management
        if (this.papers.length < 3) {
            this.fetchPapers(); 
        }
        
        // Return null if empty (async wait loop handled by caller or just try next frame)
        if (this.papers.length === 0) return null;
        
        return this.papers.shift();
    }
}

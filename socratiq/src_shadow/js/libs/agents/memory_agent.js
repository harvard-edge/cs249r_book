import pkg from 'flexsearch';
const { Index, Document, Worker } = pkg;

const options = "performance";

const index = new Index();

// const index_doc = new Document({
//     id: "id",
//     index: ["title", "content", "url", "processedDate"]
// });

const index_doc = new Document({
    id: "id",
    index: ["title", "content", "url", "processedDate"],
    context: true,
    tokenize: "forward", // Tokenize from start to end of words
    depth: 3, // Increasing depth for better contextual awareness
    optimize: true, // Enable optimization for better performance
    resolution: 100, // Increase resolution for finer distinctions in relevance scoring
    threshold: 0, // Lower threshold to include more documents in results
    charset: 'latin:advanced' // Example of setting a charset for improved text processing
});

// const index = new FlexSearch.Index({
//     encode: "extra", // Use a strong encoder
//     tokenize: "forward", // Tokenize from start to end of words
//     threshold: 0, // Lower threshold to include more documents in results
//     resolution: 100 // Higher resolution for finer score distinctions
// });


// create a way to add to each document

// const document = new Document(options);
// const worker = new Worker(options);

// const documentOptions = {
//     ...indexOptions,
//     document: {
//         id: "id",
//         index: ["title", "content"]
//     }
// };
// const documentIndex = new Document(documentOptions);

// Create a FlexSearch index
// var index = new Index("performance");


let conversations = [];
let nextId = 1;

// Function to add a new document to FlexSearch
export function addDocumentToFlexSearch(url, text) {
    const id = nextId++; // Generate a new ID
    const newDocument = { id: String(id), url, text, processedDate: new Date().toISOString() };
    conversations.push(newDocument);

    // Add document to FlexSearch index
    index.add(id, `${url} ${text}`); // Combining URL and text for indexing
    return newDocument;
}

// Function to search the top K documents in FlexSearch
export function searchDocumentsFlex(query, k = 5) {
    const searchResults = index.search({
        query: query,
        limit: k
    });

    // Map search results to retrieve full document details
    const topKDocuments = searchResults.map(result => conversations.find(conv => conv.id === String(result)));
    return topKDocuments;
}



// Optional: Pre-load documents into the index on startup or when needed
function preloadDocuments() {
    conversations.forEach(doc => {
        index.add(doc.id, `${doc.url} ${doc.text}`);
    });
}


// //////////////////////// with docs ////////////////////////

let conversations_doc = [];
let nextId_doc = 1;

// Adding documents to a document index
export function addDocumentsToFlexSearch_doc(documents) {

    documents.forEach(doc => {
        const id = nextId_doc++;  // Generate a new ID
        const newDocument = {
            id: String(id),
            url: doc.url,
            content: doc.text,
            processedDate: doc.processedDate
        };

        conversations_doc.push(newDocument);
        index_doc.add(newDocument);  // Make sure this is correct
    });

}

// Function to search the top K documents in FlexSearch
export function searchDocumentsFlex_doc(query, k = 5) {
    const searchResults = index_doc.search({
        query: query,
        limit: k
    });

    // Flatten the result IDs and remove duplicates
    const resultIds = new Set();
    searchResults.forEach(item => {
        item.result.forEach(id => resultIds.add(id));
    });

    // Find and return the documents corresponding to the result IDs
    const topKDocuments = Array.from(resultIds).map(id => {
        return conversations_doc.find(conv => conv.id === id);
    }).filter(doc => doc !== undefined);  // Filter out any undefined entries if no document matches an ID

    return topKDocuments;
}

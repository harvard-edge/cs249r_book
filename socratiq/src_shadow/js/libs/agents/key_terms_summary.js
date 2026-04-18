import nlp from 'compromise';

// Assuming extractKeyTerms is defined in the same module
export function extractKeyTerms(text) {
    const doc = nlp(text);
    const topics = doc.topics().out('array');
    return topics;
}


export function summarizeText(text) {
  let doc = nlp(text);
  let scoredSentences = [];

  doc.sentences().forEach(sentence => {
      let score = 0;
      let textContent = sentence.text();

      // Score sentences containing named entities more highly
      if (sentence.match('#ProperNoun').found) {
          score += 2;
      }

      // Score sentences containing numbers
      if (sentence.numbers().found) {
          score += 1;
      }

      // Optionally, add weight for longer sentences, assuming they carry more information
      if (textContent.split(' ').length > 12) { // arbitrary length threshold
          score += 1;
      }

      // Collect sentence and score
      if (score > 0) { // Only consider sentences with a positive score
          scoredSentences.push({ sentence: textContent, score });
      }
  });

  // Sort sentences by score in descending order
  scoredSentences.sort((a, b) => b.score - a.score);

  // Select top N sentences, here we choose top 3 for brevity
  let topSentences = scoredSentences.slice(0, 3).map(item => item.sentence);

  return topSentences.join(' ');
}


export function generateSummaryMarkdown(text) {
    const doc = nlp(text);
    const sentences = doc.sentences().out('array');
    let summary = [];
    let topics = extractKeyTerms(text);

    topics.forEach(topic => {
        sentences.forEach(sentence => {
            if (sentence.toLowerCase().includes(topic.toLowerCase())) {
                summary.push(sentence);
            }
        });
    });

    // Remove duplicates
    summary = [...new Set(summary)];

    // Format as Markdown
    let markdownSummary = "The key points to this article are:\n\n";
    summary.forEach(point => {
        markdownSummary += `* ${point}\n`;
    });

    return markdownSummary;
}


export function extractNouns(text) {
    const doc = nlp(text);
    const nouns = doc.nouns().out('array');
    return nouns;
  }


//   export function extractPhrasesInBrackets(text) {
//     // Regular expression to match content inside brackets
//     const regex = /\[([^\]]+)\]/g;
    
//     // Array to hold all matches
//     let matches = [];
    
//     // Variable to store the current match
//     let match;
  
//     // Use a loop to extract all matches
//     while ((match = regex.exec(text)) !== null) {
//       // The first group (match[1]) contains the content inside the brackets
//       matches.push(match[1]);
//     }
    
//     return matches;
//   }
// accumulatedResponse: **Sure, here are the 3 main keywords extracted from the text:**<br/><br/>**[Multitasking], [Podcasts], [Burnout]**
export function extractPhrasesInBrackets(text) {
    // Regular expression to match content inside brackets only
    const regex = /\[([^\]]+)\]/g;
    
    let matches = [];
    let match;

    // Find matches and push them to the matches array
    while ((match = regex.exec(text)) !== null) {
        // Since we're only looking for a single pattern now,
        // we can directly access match[1] which contains the content inside brackets
        matches.push(match[1]);
    }
    
    return matches;
}



export function extractWords(inputString) {
    // Step 1: Remove HTML tags and non-alphabetic characters, except spaces and dashes
    const cleanedString = inputString.replace(/<[^>]*>/g, "").replace(/[^a-zA-Z\s-]/g, "");
  
    // Step 2 and 3: Split by spaces or dashes and filter out any empty strings
    const wordsArray = cleanedString.split(/[\s-]+/).filter(word => word.trim() !== "");
  
    // Optional Step 4: Convert all words to lowercase (or you could use .toUpperCase() for uppercase)
    const lowercaseWordsArray = wordsArray.map(word => word.toLowerCase());
  
    return lowercaseWordsArray;
  }


  
  function findSentencesByTopicAndFilterOriginal(text, topic) {
    let sentences = nlp(text).sentences().out('array');
    let filteredSentences = sentences.filter(sentence => nlp(sentence).match(topic).found);
    // Create a copy of the original text to modify
    let modifiedText = text;
  
    // Remove the filtered sentences from the original text
    filteredSentences.forEach(sentence => {
      modifiedText = modifiedText.replace(sentence, '');
    });
  
    // Optional: Clean up any resulting double spaces or spaces before punctuation
    modifiedText = modifiedText.replace(/\s+/g, ' ').replace(/\s([,.!?])/g, '$1').trim();
  
    return { filteredSentences, modifiedText };
  }


function removeStopWords(text) {
    // Use the 'nlp' function from the compromise library to process the text
    const doc = nlp(text);
  
    // Remove stop words from the document
    const cleanedText = doc.normalize().remove('#Stop').out('text');
  
    return cleanedText;
  }
  
  function cleanText(inputText) {
    // Step 1: Remove HTML tags
    let cleanedText = inputText.replace(/<[^>]*>/g, "");
  
    // Step 2: Remove specific phrases, case-insensitive
    const phrasesToRemove = [
      "CONVERSATION HISTORY",
      "BACKGROUND KNOWLEDGE",
      "QUESTION",
      "QUOTE",
      "COMMENTS",
      "POWER_UP",
      "UNDERSTANDING"
    ];
    
    phrasesToRemove.forEach(phrase => {
      const regex = new RegExp(phrase + ":[^\\n]*\\n?", "gi");
      cleanedText = cleanedText.replace(regex, "");
    });
  
    // Optional Step 3: Trim leading and trailing spaces and replace multiple spaces with a single space
    cleanedText = cleanedText.trim().replace(/\s+/g, ' ');
  
    return cleanedText;
  }

  function cleanSpacesText(input) {
    // This will replace multiple spaces with a single space and multiple newlines with a single newline
    return input.replace(/\s+/g, ' ').replace(/\n+/g, '\n').trim();
  }
 
  function extractInnerText(htmlString) {
    // Create a new div element
    const tempDiv = document.createElement('div');
    
    // Assign the HTML content to the div
    tempDiv.innerHTML = htmlString;
    
    // Use the textContent property to get the text inside the div
    return tempDiv.textContent || tempDiv.innerText || "";
  }

  export function getTopics(text){
    let cleaned = extractInnerText(text) // is this slow

     cleaned = cleanText(text)

    cleaned = cleanSpacesText(cleaned)

    cleaned = removeStopWords(cleaned)

    let topics = extractKeyTerms(cleaned)

    if (topics.length === 0){
        topics = extractNouns(cleaned)

        if(topics.length === 0){
         topics = cleaned.split(" ")
        }
    }

    return topics
  }


/**
 * Finds relevant sentences in the query based on the conversation history.
 *
 * @param {string} query - The input query to search for relevant sentences.
 * @param {string} conversationHistory - The history of conversations to search for relevant sentences.
 * @return {object} An object containing the relevant sentences and the final cleaned conversation history.
 */
export function findRelevantSentencesInQuery(query, conversationHistory) {
    let queryTopics = extractKeyTerms(query); // Assuming this function is defined elsewhere
    if(queryTopics.length === 0){
        queryTopics = extractNouns(query);
        if(queryTopics.length === 0){
            queryTopics = removeStopWords(query)
        }
    }
    

    let cleanedConvoHistory = cleanText(conversationHistory); // Clean the conversation history
    let keySentences = [];

    if(queryTopics.length === 0){
      return []
    
    }
    queryTopics.forEach(topic => {
      // Find sentences relevant to the topic and get the modified text
      const result = findSentencesByTopicAndFilterOriginal(cleanedConvoHistory, topic);
      cleanedConvoHistory = result.modifiedText; // Update the conversation history without the found sentences
      keySentences = [...keySentences, ...result.filteredSentences]; // Collect all relevant sentences
    });
  
    return  keySentences; // Return both the relevant sentences and the final cleaned text
  }
  
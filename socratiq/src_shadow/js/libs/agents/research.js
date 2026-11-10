// const baseUrl = 'http://export.arxiv.org/api/query?search_query=';
const baseUrl = 'https://export.arxiv.org/api/query?search_query=';
// import {getIndicesBetQueryTextsInChunks} from "./embeddings_agent.js";
// import { alert } from "../utils/utils.js";


  // Function to fetch data from arXiv for a list of search terms
  export async function fetchArxivDataForTerms(searchTerms, num_papers) {
    // Maps each search term to a promise that resolves to the fetched data
    const promises = searchTerms.map(async (term) => {
      // Adjust the URL to include the num_papers parameter
      const response = await fetch(`${baseUrl}${term}&max_results=${num_papers}`);
      const data = await response.text();
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(data, 'text/xml');
      
      // Extract data from the XML document
      return Array.from(xmlDoc.querySelectorAll('entry')).map(entry => {
        const title = entry.querySelector('title').textContent;
        const link = entry.querySelector('id').textContent;
        const author = entry.querySelector('author name').textContent;
        const summary = extractFirstTwoSentences(entry.querySelector('summary').textContent);
        const published = entry.querySelector('published').textContent;
        const year = new Date(published).getFullYear();
        
        return { title, link, author, summary, year };
      });
    });
  
    // Wait for all promises to resolve and flatten the results
    return (await Promise.all(promises)).flat();
  }


   // Assumes `fetchArxivDataForTerms` is defined elsewhere

  //   function formatResultsToMarkdown(results) {
  //   let markdownString = '';
  
  //   results.forEach((result, index) => {
  //     // Header for each paper
  //     markdownString += `### ${index + 1}. ${result.title}\n\n`;
  //     // Author(s)
  //     markdownString += `**Author(s):** ${result.author}\n\n`;
  //     // Link
  //     markdownString += `**Link:** [Read Paper](${result.link})\n\n`;
  //     // Summary

  //     const quote_in_markdown = `\n::: spoiler Summary\n*${result.summary}*\n:::\n\n\n\n`;

  //     markdownString += quote_in_markdown //`**Summary:** ${result.summary}\n\n`;
  //     // Separator between papers
  //     markdownString += '---\n\n';
  //   });
  
  //   return markdownString;
  // }

function removeNewlines(str) {
  return str.replace(/[\r\n]+/g, '');
}



function formatResultsToMarkdown(randomID,results) {
  let markdownString = '';

  // Limit the initial display to 5 papers
  const initialDisplayLimit = 5;

  results.slice(0, initialDisplayLimit).forEach((result, index) => {
    markdownString += formatSinglePaper(result, index);
  });

  if (results.length > initialDisplayLimit) {
    const randomID_str = `more-papers-${randomID}`; // Generate a unique ID using `generateUniqueString()`
//     markdownString += `
// <button onclick="${shadowDom}.querySelector('.${randomID}').style.display='block'; this.style.display='none';">Show More Papers</button>
// <div id="more-papers" class="${randomID}" style="display: none;">`;

markdownString += `

<button id="button-more-papers-${randomID}" class="button-more-papers-${randomID} bg-white text-black border border-blue-500 px-4 py-2 rounded">Show More Papers</button>
<div id="${randomID_str}" class="${randomID_str}" style="display: none;">

`;

    results.slice(initialDisplayLimit).forEach((result, index) => {
      markdownString += formatSinglePaper(result, index + initialDisplayLimit);
    });

    markdownString += `<div class="flex flex-row justify-center gap-2 items-center mb-4">
<input id="more-papers-search-${randomID}" type="text" placeholder="Expand your search" class="w-64 px-4 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" />
<button id="more-papers-search-button-${randomID}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-r-md">Search</button>
</div></div>`;
  }

  // console.log("markdownString", markdownString);
  return markdownString;
}

function formatSinglePaper(result, index) {
  let paperString = '';

  paperString += `|  Title  |Link |\n`;
  paperString += `|---------|-----|\n`;

  paperString += `| ${removeNewlines(result.title)} | <a href="${result.link}" style="color:var(--socratiq-accent, #8e1537);" target="_blank" rel="noopener noreferrer">↗Link</a> |\n\n`;
  
  if (result.year) {
    paperString += `⭈ Year: ${result.year}\n\n`;
  }

  if (result.citations && result.citations !== 'Not available') {
    paperString += `☆ Citations: ${result.citations}\n\n`;
  }
  // const roundedNumber = parseFloat(number.toFixed(2)); // Rounds to 2 decimal places and converts back to number
// console.log(roundedNumber); // Output: 3.14

  if(result.similarity){
    paperString += `≈ Similarity: ${(result.similarity*100).toFixed(2)}%\n\n`;
  }

  const quote_in_markdown = `::: spoiler Summary\n*${result.summary}*\n:::\n\n`;

  paperString += quote_in_markdown;
  paperString += '<br/><hr/><br/>\n\n';

  return paperString;
}


// function formatResultsToMarkdown(results) {
//     let markdownString = '';
  
//     results.forEach((result, index) => {

//       // console.log("result.title", result.title)
//       // Add a small header for each paper with its number
//       // markdownString += `### Paper ${index + 1}\n\n`;
      
//       // // Start table for title, authors, and link
//       // markdownString += `| Title | Author(s) | Link |\n`;
//       // markdownString += `|-------|-----------|------|\n`;

//             // Start table for title, authors, and link
//             markdownString += `|  Title  |Link |\n`;
//             markdownString += `|---------|-----|\n`;
      
//       // Populate table with data
//       markdownString += `| ${removeNewlines(result.title)} | <a href="${result.link}" style="color:blue; hover:purple;" target="_blank">↗Link</a> |\n\n`;
//       if(results.year){
//         markdownString += `⭈ Year: ${results.citations}\n\n`;
//       }
      
//       if(results.citations && results.citations !== 'Not available' ){
//         markdownString += `☆ Citations: ${results.citations}\n\n`;
//       }
     


//       // Summary below the table
//       const quote_in_markdown = `\::: spoiler Summary\n*${result.summary}*\n:::\n\n`;

//       markdownString +=  quote_in_markdown //`**Summary:** ${result.summary}\n\n`;
      
//       // More space between papers
//       markdownString += '<br/><br/>\n\n';
//     });
  
//     return markdownString;
//   }
  
  
  // Example usage
export async function initiateResearch(query, token, randomID, searchTerms, num_papers=10){
// async function initiateResearch(searchTerms, num_papers=3){

  // const markdownResults = await fetchArxivDataForTerms(searchTerms, num_papers).then(data => {
  const dataPapers = (await fetchPapers(searchTerms, num_papers))
    // const researchTexts = data.map(paper => paper.summary);
    console.log("dataPapers", dataPapers)

    let organizedData = dataPapers;
    // try{
    // //  organizedData =  (await getIndicesBetQueryTextsInChunks(query, dataPapers,token, num_papers, 'summary', false)) //.slice(0, num_papers);
    // }
    // catch(e) {
    //   console.log("ERRER Indexing Papers", e)
    //   organizedData = dataPapers
    // }

    // console.log("fetchPapers", dataPapers)
    // console.log("getIndicesBetQueryTexts", organizedData)

    const markdownResults = formatResultsToMarkdown(randomID, organizedData);
    // console.log(markdownResults);
    return markdownResults;
    // You can then display markdownResults in your application or webpage
}

// ADDITIONAL RESEARCH AGENTS TOOLS ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


const SEMANTIC_SCHOLAR_API_URL = 'https://api.semanticscholar.org/graph/v1/paper/search';

async function fetchPapersFromSemanticScholar_sort_by_citation(topic) {
  const url = `${SEMANTIC_SCHOLAR_API_URL}?query=${encodeURIComponent(topic)}&limit=15&fields=title,year,authors,url,abstract,citationCount`;

  const response = await fetch(url, {
    headers: {
      'Accept': 'application/json'
    }
  });
  
  if (!response.ok) {
    throw new Error(`Semantic Scholar API request failed with status ${response.status}`);
  }

  const data = await response.json();
  if (!data.data) {
    throw new Error('Unexpected response structure from Semantic Scholar', data);
  }

  const sortedPapers = data.data
    .map(paper => ({
      title: paper.title,
      author: paper.authors.map(author => author.name).join(', '),
      year: paper.year,
      link: paper.url,
      summary: extractFirstTwoSentences(paper.abstract) || 'No summary available',
      citations: paper.citationCount || 'Not available'
    }))
    .sort((a, b) => b.citations - a.citations);

  return sortedPapers;
}

function extractFirstTwoSentences(abstract) {
  if (!abstract) return '';
  const sentences = abstract.match(/[^\.!\?]+[\.!\?]+/g);
  if (!sentences || sentences.length === 0) return '';
  return sentences.slice(0, 2).join(' ');
}

export async function fetchPapers(topics, num_papers=5) {
  const results = [];
  const searchTerms = Array.isArray(topics) ? topics : [topics];

  try {
    const semanticScholarPapers = await fetchPapersFromSemanticScholar_sort_by_citation(searchTerms.join(', '));
    results.push(...semanticScholarPapers);
  } catch (error) {
    console.error(`Taking a little longer than expected`);
    try {
      const [arxivPapers] = await Promise.all([
        fetchArxivDataForTerms(searchTerms, num_papers),
      ]);
      results.push(...arxivPapers);
    } catch (error) {
      console.error(`Error fetching papers for topic "${JSON.stringify(topics)}":`, error);
    }
  }

  return results;
}

// Example usage
// const topics = ['cancer', 'machine learning'];
// fetchPapers(topics).then(results => {
//   console.log(JSON.stringify(results, null, 2));
// }).catch(error => {
//   console.error('Error fetching papers:', error);
// });

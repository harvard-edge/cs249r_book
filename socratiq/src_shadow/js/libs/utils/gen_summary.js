export function extractHeadingsAndText(maxCharacters = 600) {
    // Select all heading elements on the page, if none, select paragraph elements
    let elements = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    if (elements.length === 0) {
      elements = document.querySelectorAll('p');
    }
    
    let extractedText = '';
    
    elements.forEach(element => {
      let headingText = `${element.tagName}: ${element.textContent}\n`;
      let sentencesFound = 0;
      let nextElement = element.nextElementSibling;
      
      while (nextElement && sentencesFound < 2) {
        const textContent = nextElement.textContent;
        const sentences = textContent.match(/[^\.!\?]+[\.!\?]+/g) || [];
        
        sentences.some(sentence => {
          // Check if adding this sentence would exceed the maxCharacters limit
          if (maxCharacters !== null && (extractedText.length + headingText.length + sentence.length) > maxCharacters) {
            return true; // Break the loop early if we exceed the maxCharacters limit
          }
          
          headingText += `${sentence.trim()}\n`;
          sentencesFound++;
          return sentencesFound >= 2;
        });
        
        nextElement = sentencesFound < 2 ? nextElement.nextElementSibling : null;
      }
      
      if (maxCharacters === null || (extractedText.length + headingText.length) <= maxCharacters) {
        extractedText += headingText + '\n';
      } else {
        // If adding the next section would exceed the limit, stop the function
        return false; // Break the forEach loop
      }
    });
    
    return extractedText;
  }
  
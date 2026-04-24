
import { debounce } from "../../../libs/utils/utils";
import { suggestion_agent } from './highlight_menu';

let suggestionOverlay;
let suggestion = "";
let lastInput = "";
let suggestions = ['what does this mean?', 'give real life examples of this'];
let matchFound;


function truncateString(str, num) {
    if (str.length <= num) {
        return str;
    } else {
        return str.slice(0, num) + '...';
    }
}


function createElementFromHTML(htmlString) {
    const div = document.createElement('div');
    div.innerHTML = htmlString.trim();
    return div.firstChild;
}

const tabHtml = `<span id="tabPlaceholder" class="ml-1 bg-white text-black text-xs font-semibold opacity-50 px-2 py-1" style="border: 1px solid black; padding: 4px;">TAB</span>`;
const tabElement = createElementFromHTML(tabHtml);

function retainSpecificChars(str) {
    return str.replace(/[^a-zA-Z0-9 \-+?.,:;"'*()&^%$#@!{}\[\]]/g, '');
}

function clearSuggestion() {
    suggestion = "";
    suggestionOverlay.innerHTML = "";
}

export function text_area_suggestions(shadowEle) {
    suggestionOverlay = shadowEle.querySelector('#suggestionOverlay');
    const textarea = shadowEle.querySelector('#responseTextarea');

    textarea.addEventListener('input', function(e) {
        displaySuggestion(e.target.value, shadowEle);
    });

    textarea.addEventListener('focus', function() {
        if (this.value === '') {
            // If there's no input, display a random suggestion
            displaySuggestion('', shadowEle);
        }
        this.removeAttribute('placeholder');
    });

    textarea.addEventListener('blur', function() {
        if (this.value.length === 0) {
            clearSuggestion();
            this.setAttribute('placeholder', 'Write something...');
        }
    });

    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Tab' && suggestion) {
            e.preventDefault();
            acceptSuggestion(shadowEle);
        } else if (e.key === 'Backspace' && suggestion) {
            clearSuggestion();
        }
    });
    function displaySuggestion(inputText, shadowEle) {
        // When the input changes, call the suggestion agent
        debouncedCallSuggestAgent(inputText);
    
        let lowercaseInput = inputText.toLowerCase();
        lowercaseInput = retainSpecificChars(lowercaseInput);
        matchFound = false;
    
        if (inputText) {
            // Find a suggestion that starts with the current input
            matchFound = suggestions.some(sugg => {
                if (sugg.toLowerCase().startsWith(lowercaseInput)) {
                    suggestion = sugg.substring(inputText.length);
                    suggestionOverlay.textContent = truncateString(inputText + suggestion, 60)
                    suggestionOverlay.appendChild(tabElement);
                    return true;
                }
                return false;
            });
        } else if (inputText.length === 0) {
            // console.log("RANDOM SUGGESTION SHOWING")
            // If textarea is focused but empty, show a random suggestion
            const randomIndex = Math.floor(Math.random() * suggestions.length);
            suggestion = suggestions[randomIndex];
            suggestionOverlay.textContent = truncateString(suggestion, 60) // + suggestion;
            suggestionOverlay.appendChild(tabElement);
        }
        
        // console.log("I AM INPUTTEXT", inputText)
        // If no match was found, clear the suggestion overlay
        if (!matchFound && inputText.length > 0) {
            // console.log("NO MATCH FOUND", inputText)
            clearSuggestion();
        }
    }
    
    textarea.addEventListener('focus', function() {
        displaySuggestion('', shadowEle); // Directly display a suggestion on focus
        this.removeAttribute('placeholder');
    });
    

    function acceptSuggestion(shadowEle) {
        const textarea = shadowEle.querySelector('#responseTextarea');
        if (suggestion) {
            textarea.value += suggestion; // Append the suggestion
            displaySuggestion(textarea.value, shadowEle); // Update overlay
            clearSuggestion();
        }
    }
}

async function callSuggestAgent(currentInput) {
    if (currentInput !== lastInput && currentInput) {
        lastInput = currentInput;
        const prompt = `\nGiven a piece of text and a partial query as input: <${lastInput}> predict the completion of the query inside the <>. Output only the completed query.`;
        const newSuggestion = await suggestion_agent(prompt);
        
        if (newSuggestion) {
            suggestions.unshift(retainSpecificChars(newSuggestion));
            suggestions = suggestions.slice(0, 5); // Keep only the latest 5 suggestions
        }
    }
}

const debouncedCallSuggestAgent = debounce((inputText) => callSuggestAgent(inputText), 500);

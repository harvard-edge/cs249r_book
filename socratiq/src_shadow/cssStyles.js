export const styles = `

/* SIMPLE THEME SYSTEM - NO !important */
:host,
.socratiq-widget-root {
    --socratiq-text: #1f2328; /* Light theme default: dark text */
    --socratiq-bg: #ffffff;
    --socratiq-muted: #4c566a;
    --socratiq-border: #d0d7de;
    --socratiq-hover: #f6f8fa;
    --socratiq-accent: #0969da;
    --socratiq-accent-bg: #dbeafe;
    --socratiq-accent-hover: #bfdbfe;
    color: var(--socratiq-text);
    background: var(--socratiq-bg);
}

:host([data-socratiq-theme="dark"]),
.socratiq-widget-root[data-socratiq-theme="dark"] {
    --socratiq-text: #e6edf3; /* Dark theme: light text */
    --socratiq-bg: #0d1117;
    --socratiq-muted: #9ca3af;
    --socratiq-border: #30363d;
    --socratiq-hover: #21262d;
    --socratiq-accent: #58a6ff;
    --socratiq-accent-bg: #1f2937;
    --socratiq-accent-hover: #374151;
}

#text-selection-menu{
    z-index: 9999;
    position: fixed;
  }
  
  
  /* If the icons class is for SVGs within the buttons, adjust their size here */
  .icons {
      width: 24px; /* Adjust SVG size as needed */
      height: 24px; /* Adjust SVG size as needed */
      margin: auto;
  }
  .icon-tiny
  {
      width: 12px; /* Adjust SVG size as needed */
      height: 12px; /* Adjust SVG size as needed */
  }
  
  .icons-left {
    width: 24px; /* Adjust SVG size as needed */
    height: 24px; /* Adjust SVG size as needed */
  }
  /*  */
  
  /* Basic table styles */
  table {
    width: 100%; /* Full-width */
    border-collapse: collapse; /* Collapse borders */
    margin: 20px 0; /* Add some margin around the table */
    font-size: 0.9em; /* Adjust font size */
    font-family: Arial, sans-serif; /* Use a nice font-family */
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); /* Slight shadow around table */
  }
  
  /* Table headers */
  th {
    background-color: #009879; /* A nice green background */
    color: #ffffff; /* White text color */
    text-align: left; /* Align text to the left */
    padding: 12px 15px; /* Add some padding */
  }
  
  /* Table cells */
  td {
    padding: 12px 15px; /* Add some padding */
    border-bottom: 1px solid #dddddd; /* A light border for each cell */
  }
  
  /* Table row hover effect */
  tr:nth-of-type(even) {
    background-color: #f3f3f3; /* Light grey background for even rows */
  }
  
  tr:hover {
    background-color: #f1f1f1; /* Slightly different grey for hover effect */
  }
  
  /* Responsive tables */
  @media screen and (max-width: 600px) {
    table {
      width: 100%;
      display: block;
      overflow-x: auto; /* Enable horizontal scrolling on small devices */
    }
  }
  
  
  /* iframe */
  
  /* body {
      margin: 0;
      padding: 0;
  } */
  
  /* .iframe-container {
      position: relative;
      background-color: transparent ;
      max-width: 200px;
      border: none;
      max-height: fit-content;
      overflow: hidden;
      border: 2px solid black;
      overflow: visible;
      max-height: 300px; 
      overflow-y: auto; 
      box-sizing: border-box;
      background-color: #fff;
  
  } */
  
  
  
  #markdown-preview {
      /* min-width: 50px; */
      width: 100%;
      background-color: #fff;
      overflow: visible; /* Let it expand */ /* hide any overflow initially */
      transition: height 0.3s ease-out, opacity 0.3s ease-out; /* Add opacity to the transition *//* animate height changes */
      padding: 4px;
      word-wrap: break-word;
      word-break: break-word;
      overflow-wrap: break-word;
  }
  
  /* Force proper text wrapping for pre elements */
  #markdown-preview pre, #markdown-preview pre.dark-mode {
      white-space: pre-wrap !important;
      word-wrap: break-word !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
      max-width: 100% !important;
  }
  
  #markdown-preview pre[style*="white-space: normal"] {
      white-space: pre-wrap !important;
  }
  
  /* Override any inline white-space styles on pre elements */
  #markdown-preview pre[style*="white-space"] {
      white-space: pre-wrap !important;
  }
  
  /* Ensure code elements inside pre also wrap properly */
  #markdown-preview pre code {
      white-space: pre-wrap !important;
      word-wrap: break-word !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
  }
  
  /* Handle dark-mode pre elements specifically */
  #markdown-preview pre.dark-mode, #markdown-preview pre.dark-mode code {
      white-space: pre-wrap !important;
      word-wrap: break-word !important;
      word-break: break-word !important;
      overflow-wrap: break-word !important;
      max-width: 100% !important;
  }
    
    .iframe-container .markdown-preview {
      border: none;
      max-height: 300px; /* set maximum height */
      overflow-y: auto; /* show scrollbars when needed */
    }
  
    
  
  
  
  
  
  .button-row{
      display: flex;
      justify-content: last baseline;
      gap: 10px;
      flex-direction: row;
      background-color: transparent;
  }
  
  #quiz-form {
      margin-bottom: 1rem; /* Tailwind class mb-4 */
    }
    
    #submit-quiz-btn {
      margin-top: 1rem; /* Tailwind class mt-4 */
      width: 100%; /* Tailwind class w-full */
      background-color: black; /* Tailwind class bg-blue-500 */
      color: white; /* Tailwind class text-white */
      padding: 0.5rem 1rem; /* Tailwind classes px-4 py-2 */
      border-radius: 0.25rem; /* Tailwind class rounded */
      transition: background-color 0.2s; /* Tailwind hover state */
      margin-bottom: 10px;
    }
    
    #submit-quiz-btn:hover {
      background-color: #2779bd; /* Tailwind class hover:bg-blue-600 */
    }
    
    .submit-quiz-btn {
      margin-top: 1rem; /* Tailwind class mt-4 */
      width: 100%; /* Tailwind class w-full */
      background-color: black; /* Tailwind class bg-blue-500 */
      color: white; /* Tailwind class text-white */
      padding: 0.5rem 1rem; /* Tailwind classes px-4 py-2 */
      border-radius: 0.25rem; /* Tailwind class rounded */
      transition: background-color 0.2s; /* Tailwind hover state */
      margin-bottom: 10px;
    }
    
    .submit-quiz-btn:hover {
      background-color: #2779bd; /* Tailwind class hover:bg-blue-600 */
    }
    
  
    #quiz form{
      display: block;
      /* flex-direction: column; */
      box-shadow: 0 0 0px 0 rgba(0, 0, 0, 0.1);
    }
  
    #quiz h4 {
      font-size: 16px; /* Change this value to adjust the font size */
    }
  
   #quiz ul {
      list-style-type: none;
    }
  
    #result {
      margin-top: 1rem; /* Tailwind class mt-4 */
      text-align: center; /* Tailwind class text-center */
    }
    
    details > .spoiler-content,
    details > p,
    details > em {
      color: black;
      font-style: normal !important; /* This will override other styles unless they also use !important */
    }
  /*   
    .correct-answer::after {
      content: '✅';
      color: green;
      margin-right: 8px;
  } */
  .wrong-answer::after {
      content: '❌';
      color: #f87171;
      margin-left: 8px;
  }
  
  
  /* Style adjustments */
   .icon-button {
      cursor: pointer;
      padding: 10px 20px;
      background-color: #000; /* Make buttons black */
      color: #fff; /* Text color white for contrast */
      border: none;
      border-radius: 5px;
      margin: 20px;
    }
  /* ///////////////////////////////////////////////////////////////////// */
      /* General styles for the slider */
    input[type=range] {
      -webkit-appearance: none; /* Override default CSS styles */
      appearance: none;
      width: 100%; /* Full-width */
      height: 5px; /* Specified height */
      background: black; /* Black track */
      outline: none; /* Remove outline */
      opacity: 0.7; /* Partial transparency */
      transition: opacity .2s; /* Transition for the slider */
    }
    
  /* Initially hide the popup div */
  
  /* Show the popup div when hovering over the button or the popup div itself */
  .hover-btn:hover + .popup-content, .popup-content:hover {
    visibility: visible;
    opacity: 1;
    transition-delay: 0s; /* Make popup appear immediately */
  }
    
    /* Style for Webkit browsers like Chrome, Safari */
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; /* Override default CSS styles */
      appearance: none;
      width: 25px; /* Width of the thumb */
      height: 25px; /* Height of the thumb */
      background: black; /* Black thumb */
      cursor: pointer; /* Cursor on hover */
    }
  
    /* Style for Mozilla Firefox */
    input[type=range]::-moz-range-thumb {
      width: 25px; /* Width of the thumb */
      height: 25px; /* Height of the thumb */
      background: black; /* Black thumb */
      cursor: pointer; /* Cursor on hover */
    }
  
    /* Style for the focus state */
    input[type=range]:focus {
      outline: none; /* Remove the outline */
    }
    
    /* Additional style for the focus state on Webkit browsers */
    input[type=range]:focus::-webkit-slider-thumb {
      background: #333; /* Darker shade when focused */
    }
  
    /* Additional style for the focus state on Mozilla Firefox */
    input[type=range]:focus::-moz-range-thumb {
      background: #333; /* Darker shade when focused */
    }
  
    /* Show the popup div on hover */
    .hover-btn:hover + .popup-content, .popup-content:hover {
      display: block;
    }
  
    /* Style for the slider */
    .slider {
      width: 100%;
      margin: 20px 0;
      
    }
  
    /* Slider labels */
    .slider-labels {
      display: flex;
      justify-content: space-between;
    }
  
    /* Style adjustments for checkboxes */
    .checkbox-container {
      margin: 10px 0;
    }
  
    /* Making the icons black, assuming SVG icons */
    .icon-button img {
      filter: brightness(0) invert(0); /* A general way to make images black */
    }
  
    .icon-button {
      color: white;
      margin: auto;
    }
  
    .button-and-popup-container {
      position: relative;
      display: inline-block; /* Or any other display type that suits your layout */
    }
    
    .settings-button:hover + .popup-content,
  .popup-content:hover {
    visibility: visible; /* Show the popup */
    opacity: 1;
    transition: opacity 0.5s ease;
    transition-delay: 0s; /* React immediately when hovered */
  }
  
  
  /* Initially hide the modal */
  .modal {
    display: none;
    position: fixed;
    z-index: 100;
    left: 50%;
    top: 50%;
    padding: 10px;
    width: 400px;
    height: fit-content;
    transform: translate(-50%, -50%); /* Center the modal */
    /* overflow: auto; */
    /* background-color: rgba(0, 0, 0, 0.4); */
  }
  
  /* #popover-container {
    position: fixed;  
    z-index: 1000;  
    top: 50%;       
    left: 50%;
    transform: translate(-50%, -50%);  
  } */
  
  
  
  #popover-container .overflow-auto {
    max-height: 250px;
    overflow: auto;
  }
  
  /* Modal Content */
  .modal-content {
    background-color: #fefefe;
    margin: 15% auto;
    padding: 20px;
    /* border: 1px solid #888; */
    width: 80%;
  }
  
  /* The Close Button */
  .close {
    color: #aaaaaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
  }
  
  .close:hover,
  .close:focus {
    color: black;
    text-decoration: none;
    cursor: pointer;
  }
  
  
  
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeOutDown {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(20px); }
  }
  .animate-fadeInUp {
    animation: fadeInUp 0.5s ease-out forwards;
  }
  .animate-fadeOutDown {
    animation: fadeOutDown 0.5s ease-out forwards;
  }
  
  .rotate-180 {
    transform: rotate(180deg);
  }
  
  /* 
  #enter-btn {
    top: -0.3rem; 
  } */

    
    #enter-btn {
  transition: transform 0.2s ease, color 0.2s ease;
}
  
#enter-btn:hover {
  transform: translateY(-50%) scale(1.1);
  color: #3b82f6; /* Tailwind's blue-500 */
}

#enter-btn svg {
  transition: transform 0.2s ease;
}

#enter-btn:hover svg {
  transform: rotate(-45deg) scale(1.1);
}
  
  
  /* HEADINGS LISTS */
  
  /* Overriding Tailwind styles with higher specificity */
    /* For headers */
    h1 {
      font-size: 2rem;
      font-weight: 700;
    }
  
    h2 {
      font-size: 1.5rem;
      font-weight: 600;
    }
  
    /* ... include your other styles ... */
  
    /* For unordered lists */
    ul {
      list-style-type: disc;
      padding-left: 1.5rem;
    }
  
    /* For ordered lists */
    ol {
      list-style-type: decimal;
      padding-left: 1.5rem;
    }
  
    /* For list items */
    li {
      margin-bottom: 0.5rem;
    }
  
    ul, ol {
      list-style-type: none; /* Removes the default list-style */
      padding: 0;           /* Removes the default padding */
      margin: 0;            /* Removes the default margin */
    }
    
  
    /* LIGHT MODE // / // DARK MODE */
    
    .dark-mode {
      /* --background-color: #1e1e1e; */
      --text-color: black;
      /* --border-color: #333; */
      --link-color: #4ea0f6;
      /* --secondary-background-color: #2a2a2a; */
  }
  
  .dark-mode {
      /* background-color: var(--background-color); */
      color: var(--text-color);
  }
  
  .dark-mode a {
      color: var(--link-color);
  }
  
  .dark-mode .secondary-background {
      /* background-color: var(--secondary-background-color); */
  }
  
  .dark-mode .bordered {
      /* border: 1px solid var(--border-color); */
  }
  
  
  .light-mode {
    --background-color: #ffffff;
    --text-color: #333333;
    --border-color: #ddd;
    --link-color: #007bff;
    --secondary-background-color: #f9f9f9;
  }
  
  .light-mode {
    background-color: var(--background-color);
    color: var(--text-color);
  }
  
  .light-mode a {
    color: var(--link-color);
  }
  
  .light-mode .secondary-background {
    background-color: var(--secondary-background-color);
  }
  
  .light-mode .bordered {
    border: 1px solid var(--border-color);
  }
  
  
  .icon-mini {
    height: 1rem;
    width: 1rem;
  }
  
  
  /* Add the CSS class for the button */
  .followup-button {
    cursor: pointer; /* Optional: Changes the mouse cursor on hover */
  }
  
  /* Define the hover state for the button */
  .followup-button:hover {
    color: purple; /* Change text color to purple on hover */
  }
  
  
  #menu-toggle:checked~label {
    background-color: #2563eb;
    /* Darker shade for active state */
  }
  

  
  #input-container {
    position: relative;
  }
  
  #copy-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 8px;
    padding: 8px;
    border-radius: 4px;
  }
  
  .input-wrapper {
    position: relative;
    width: 100%;
  }
  
  #user-input {
    width: 100%;
    padding-right: 2.5rem; /* Space for the send button */
  }
  
  #enter-btn {
    position: absolute;
    /* top: 50%; */
    top: 46%;
    transform: translateY(-50%); /* Center vertically */
    display: flex;
    align-items: center;
  }
  
  
  #copy-container.hidden {
    display: none !important; /* Hide copy-container when no items are present */
  }
  
  
  @keyframes inner-pulse {
    0% {
      box-shadow: inset 0 0 10px rgba(0, 123, 255, 0.7), inset 0 0 20px rgba(0, 123, 255, 0.5);
    }
    50% {
      box-shadow: inset 0 0 15px rgba(0, 123, 255, 1), inset 0 0 30px rgba(0, 123, 255, 0.8);
    }
    100% {
      box-shadow: inset 0 0 10px rgba(0, 123, 255, 0.7), inset 0 0 20px rgba(0, 123, 255, 0.5);
    }
  }
  
  .pulse-effect {
    animation: inner-pulse 1.5s ease-in-out infinite;
    border-radius: 6px;
  }
  
  
  #context-button {
    margin-left: 8px;
    /* context button */
    position: relative;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 8px; 
    margin-bottom: 8px;
    /* background-color: white; */
    padding: 4px;
    /* border: 1px solid #ddd;  */
    /* box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);  */
  }
  
  #context-button:hover {
    background-color: #f5f5f5; /* Change background color on hover */
    border-color: #ccc; /* Change border color on hover */
  }

  #quiz-button {
    /* quiz button */
    position: relative;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 8px; 
    margin-bottom: 8px;
    /* background-color: white; */
    padding: 4px;
    /* border: 1px solid #ddd;  */
    /* box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);  */
  }
  
  #quiz-button:hover {
    background-color: #f5f5f5; /* Change background color on hover */
    border-color: #ccc; /* Change border color on hover */
  }
  
  .big-quiz-btn-disabled {
    background-color: #666 !important;
    color: #ccc !important;
    cursor: not-allowed !important;
  }
  
  .big-quiz-btn-disabled:hover {
    background-color: #666 !important;
    color: #ccc !important;
  }
  
  /* smart input */
  
  .at-mentions-dropup {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e0 #fff;
  }
  
  .at-mentions-dropup::-webkit-scrollbar {
    width: 6px;
  }
  
  .at-mentions-dropup::-webkit-scrollbar-track {
    background: #fff;
  }
  
  .at-mentions-dropup::-webkit-scrollbar-thumb {
    background-color: #cbd5e0;
    border-radius: 3px;
  }
  
  .at-mention-item {
    transition: background-color 0.2s;
  }
  
  .hidden-mention {
    display: none !important;
    position: absolute !important;
    visibility: hidden !important;
    pointer-events: none !important;
  }
  
  .folder-filters {
    position: sticky;
    top: 0;
    background: white;
    z-index: 1;
  }
  
  .folder-filter {
    font-size: 0.875rem;
    transition: background-color 0.2s;
  }
  
  .folder-filter:hover {
    background-color: #f3f4f6;
  }
  
  .results-container {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e0 #fff;
  }
  
  .results-container::-webkit-scrollbar {
    width: 6px;
  }
  
  .results-container::-webkit-scrollbar-track {
    background: #fff;
  }
  
  .results-container::-webkit-scrollbar-thumb {
    background-color: #cbd5e0;
    border-radius: 3px;
  }
  /* ── More-options dropdown ─────────────────────────────────────────────── */
  .more-options-dropdown {
    position: fixed;
    background: var(--socratiq-bg, #ffffff);
    border: 1px solid var(--socratiq-border, #e5e7eb);
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    min-width: 190px;
    z-index: 2147483640;
    overflow: hidden;
    animation: dropdownRiseIn 0.15s ease-out;
  }
  .more-options-dropdown.hidden { display: none !important; }
  @keyframes dropdownRiseIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .dropdown-content { padding: 4px; }
  .dropdown-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 10px;
    border-radius: 7px;
    font-size: 0.82rem;
    color: var(--socratiq-text, #374151);
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.12s;
  }
  .dropdown-item:hover { background: rgba(99,102,241,0.08); color: #6366f1; }
  .dropdown-item.active {
    background: rgba(99,102,241,0.12);
    color: #6366f1;
    font-weight: 600;
  }

  /* Draw-to-select active indicator on the ⋯ button */
  #more-options-button.draw-active {
    color: #6366f1;
    background: rgba(99,102,241,0.12);
    border-radius: 6px;
  }

  .difficulty-dropdown {
    position: relative;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: flex-end;
    z-index: 50; /* Ensure it's above other content */
  }
  
  .difficulty-options {
    position: absolute;
    right: 0;
    min-width: 150px;
    border: 1px solid rgba(0,0,0,0.1);
    background-color: white;
    border-radius: 0.375rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    z-index: 9999; /* Higher z-index to ensure visibility */
    display: block;
  }
  
  .difficulty-options.hidden {
    display: none; /* Show when not hidden */
  }
  
  .difficulty-option {
    transition: background-color 0.2s;
    font-size: 0.75rem;
    cursor: pointer;
    white-space: nowrap;
  }
  
  /* Animation for dropdown */
  .difficulty-options:not(.hidden) {
    animation: dropdownFade 0.2s ease-out;
  }
  
  @keyframes dropdownFade {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  /* Add this section after your existing styles */
  
  /* Tooltip styles */
  /* Tooltip styles */
  /* .tooltip-container {
    position: fixed;
    z-index: 10000;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
  } */
  
  
  /* prev chats */
  #popover-container {
    z-index: 102;  
    top: 50%;       
    left: 50%;
    transform: translate(-50%, -50%);  
    display: none;
  }
  
  /* Custom scrollbar */
  #list-container::-webkit-scrollbar {
    width: 8px;
  }
  
  #list-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }
  
  #list-container::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
  }
  
  #list-container::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
  }
  
  .search-input {
    transition: all 0.2s ease-in-out;
  }
  
  .search-input:focus {
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }


  /* Customize scrollbar for Webkit browsers (Chrome, Safari, etc.) */
#message-container::-webkit-scrollbar {
  width: 4px; /* thin scrollbar */
  height: 0; /* hide horizontal scrollbar */
}

#message-container::-webkit-scrollbar-track {
  background: transparent; /* transparent track */
}

#message-container::-webkit-scrollbar-thumb {
  background-color: rgba(156, 163, 175, 0.2); /* light gray with opacity */
  border-radius: 20px;
  transition: background-color 0.3s ease;
}

#message-container:hover::-webkit-scrollbar-thumb {
  background-color: rgba(156, 163, 175, 0.5); /* darker on hover */
}

/* For Firefox */
#message-container {
  scrollbar-width: thin;
  scrollbar-color: #0d6efd transparent;
  overflow-x: hidden; /* hide horizontal scrollbar */
}


.mermaid-container {
    background-color: #f8f9fa;
    padding: 1em;
    border-radius: 4px;
    margin: 1em 0;
}

.mermaid-diagram {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100px;
}

.mermaid-error {
    color: red;
    padding: 1em;
    border: 1px solid red;
    margin: 1em 0;
}

.mermaid-figure {
  margin: 1.5em 0;
  padding: 1em;
  background: #f8f9fa;
  border-radius: 4px;
}

.mermaid-figure figcaption {
  margin-top: 0.5em;
  text-align: center;
  font-style: italic;
  color: #666;
}

.mermaid {
  display: flex;
  justify-content: center;
}

.ink-mde-details {
  display: none !important;
}

.sr-notification {
    background: white;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    margin: 8px;
    padding: 12px;
    min-width: 200px;
    animation: slideIn_2 0.3s ease-out;
}

.sr-notification-content {
    display: flex;
    align-items: center;
    gap: 8px;
}

.sr-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}



@keyframes slideIn_2 {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.sr-notification.success {
    background: #e6ffe6;
    border-left: 4px solid #28a745;
}

.sr-notification.error {
    background: #ffe6e6;
    border-left: 4px solid #dc3545;
}

.sr-notification.saving {
    background: #e6f3ff;
    border-left: 4px solid #0066cc;
}

.save-button-content {
    display: flex;
    justify-content: center;
    align-items: center;
}


#saveCard:disabled {
    opacity: 0.75;
    cursor: not-allowed;
}
     .boarding-highlight {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.5);
    }

    .boarding-popover {
        background: var(--socratiq-bg, #ffffff);
        color: var(--socratiq-text, #1f2328);
        border-radius: 8px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.2);
        padding: 20px;
        max-width: 300px;
    }

    .boarding-popover-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: var(--socratiq-text, #1f2328);
    }

    .boarding-popover-description {
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 15px;
        color: var(--socratiq-muted, #6b7280);
    }

    .boarding-button {
        background: #007bff;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
    }

    .boarding-button:hover {
        background: #0056b3;
    }

#floating-ai-btn .menu-item {
  display: flex !important;
  align-items: center !important;
  padding: 8px 16px !important;
  gap: 8px !important;
  cursor: pointer !important;
  transition: transform 0.2s ease !important;
}

#floating-ai-btn .menu-item:hover {
  transform: scale(1.05) !important;
}

#floating-ai-btn {
  padding: 2px !important;
  background: rgb(75, 85, 99) !important;
  border-radius: 4px !important;
  min-width: 140px !important;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

#floating-ai-btn .menu-divider {
  height: 1px !important;
  background: rgba(255, 255, 255, 0.1) !important;
  margin: 4px 0 !important;
}

/* Boarding.js specific styles */
.socratiq-onboarding .boarding-popover,
.sr-onboarding .boarding-popover {
    max-width: 300px !important;
    max-height: 80vh !important;
    overflow-y: auto !important;
    position: fixed !important;
    background: var(--socratiq-bg, #ffffff) !important;
    color: var(--socratiq-text, #1f2328) !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.2) !important;
    padding: 20px !important;
    z-index: 99999 !important;
    word-break: break-word !important;
    transform-origin: center center !important;
    transition: opacity 0.3s ease-in-out !important;
}

/* Prevent content from causing unwanted scrolling */
.socratiq-onboarding .boarding-popover-content,
.sr-onboarding .boarding-popover-content {
    max-height: calc(80vh - 100px) !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    padding-right: 5px !important;
}

/* Custom scrollbar for the popover content */
.socratiq-onboarding .boarding-popover-content::-webkit-scrollbar,
.sr-onboarding .boarding-popover-content::-webkit-scrollbar {
    width: 4px !important;
}

.socratiq-onboarding .boarding-popover-content::-webkit-scrollbar-track,
.sr-onboarding .boarding-popover-content::-webkit-scrollbar-track {
    background: transparent !important;
}

.socratiq-onboarding .boarding-popover-content::-webkit-scrollbar-thumb,
.sr-onboarding .boarding-popover-content::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 4px !important;
}

/* Ensure popovers stay within viewport bounds */
@media screen and (max-width: 768px) {
    .socratiq-onboarding .boarding-popover,
    .sr-onboarding .boarding-popover {
        max-width: 260px !important;
        font-size: 14px !important;
    }
}

/* Add these new styles */
.push-content-enabled #mybody {
  transition: margin-left 0.3s ease-in-out;
}

.push-content-enabled.menu-open #mybody {
  margin-left: -400px; /* Adjust based on your menu width */
}

/* Toggle switch styles */
.toggle-checkbox:checked {
  right: 0;
  border-color: #68D391;
}

.toggle-label {
  transition: background-color 0.2s ease;
}

.toggle-checkbox:checked + .toggle-label {
  background-color: #48BB78;
}

`

// function adjustMessageContainerWidth() {
//   const textSelectionMenu = document.querySelector('#text-selection-menu');
//   const toggleButton = document.querySelector('#toggleButton');
//   const messageContainer = document.querySelector('#message-container');
  
//   if (textSelectionMenu && toggleButton && messageContainer) {
//       const menuWidth = textSelectionMenu.offsetWidth;
//       const buttonWidth = toggleButton.offsetWidth;
//       messageContainer.style.width = `${menuWidth - buttonWidth}px`;
//   }
// }

// // Call this function:
// // 1. After the menu is created
// // 2. On window resize
// // 3. When toggle button visibility changes

// window.addEventListener('resize', adjustMessageContainerWidth);

// // Add to your initialization code
// adjustMessageContainerWidth();
import utilities from './resources/study-utilities.css?inline';
import katex from 'katex/dist/katex.min.css?inline';
import {boardingStyles} from './js/libs/utils/boarding-styles.js';

export const styles = `${utilities}\n${katex}\n${boardingStyles}

/* SIMPLE THEME SYSTEM - NO !important */
:host,
.socratiq-widget-root {
    --socratiq-brand-50: 250 241 244;
    --socratiq-brand-100: 247 230 234;
    --socratiq-brand-200: 238 208 215;
    --socratiq-brand-300: 223 170 184;
    --socratiq-brand-400: 201 116 139;
    --socratiq-brand-500: 142 21 55;
    --socratiq-brand-600: 116 19 47;
    --socratiq-brand-700: 95 14 37;
    --socratiq-brand-800: 76 11 30;
    --socratiq-brand-900: 58 9 23;
    --socratiq-brand-950: 36 5 14;
    --socratiq-text: #272329; /* Light theme default: warm dark text */
    --socratiq-bg: #ffffff;
    --socratiq-muted: #5e5157;
    --socratiq-border: #d8c5cb;
    --socratiq-hover: #fcf8f9;
    --socratiq-accent: #8e1537;
    --socratiq-accent-strong: #74132f;
    --socratiq-accent-rgb: 142, 21, 55;
    --socratiq-accent-bg: #f7e6ea;
    --socratiq-accent-hover: #eed0d7;
    color: var(--socratiq-text);
    background: var(--socratiq-bg);
}

:host([data-socratiq-volume="vol2"]),
.socratiq-widget-root[data-socratiq-volume="vol2"] {
    --socratiq-brand-50: 241 245 251;
    --socratiq-brand-100: 227 236 248;
    --socratiq-brand-200: 198 216 240;
    --socratiq-brand-300: 155 187 228;
    --socratiq-brand-400: 107 159 212;
    --socratiq-brand-500: 31 64 122;
    --socratiq-brand-600: 22 47 89;
    --socratiq-brand-700: 18 38 73;
    --socratiq-brand-800: 14 29 56;
    --socratiq-brand-900: 10 21 41;
    --socratiq-brand-950: 6 13 26;
    --socratiq-muted: #4e5968;
    --socratiq-border: #c5d1e0;
    --socratiq-hover: #f7f9fc;
    --socratiq-accent: #1f407a;
    --socratiq-accent-strong: #162f59;
    --socratiq-accent-rgb: 31, 64, 122;
    --socratiq-accent-bg: #e3ecf8;
    --socratiq-accent-hover: #c6d8f0;
}

:host([data-socratiq-volume="vol3"]),
.socratiq-widget-root[data-socratiq-volume="vol3"] {
    --socratiq-brand-50: 248 243 252;
    --socratiq-brand-100: 240 230 250;
    --socratiq-brand-200: 223 201 244;
    --socratiq-brand-300: 200 162 236;
    --socratiq-brand-400: 176 124 226;
    --socratiq-brand-500: 88 28 135;
    --socratiq-brand-600: 67 20 102;
    --socratiq-brand-700: 54 16 82;
    --socratiq-brand-800: 41 12 63;
    --socratiq-brand-900: 29 8 45;
    --socratiq-brand-950: 18 5 28;
    --socratiq-muted: #585063;
    --socratiq-border: #d2c5e0;
    --socratiq-hover: #faf8fc;
    --socratiq-accent: #581c87;
    --socratiq-accent-strong: #431466;
    --socratiq-accent-rgb: 88, 28, 135;
    --socratiq-accent-bg: #f0e6fa;
    --socratiq-accent-hover: #dfc9f4;
}

:host([data-socratiq-volume="vol4"]),
.socratiq-widget-root[data-socratiq-volume="vol4"] {
    --socratiq-brand-50: 238 250 246;
    --socratiq-brand-100: 219 242 234;
    --socratiq-brand-200: 184 228 213;
    --socratiq-brand-300: 134 208 184;
    --socratiq-brand-400: 78 184 152;
    --socratiq-brand-500: 6 95 70;
    --socratiq-brand-600: 4 70 52;
    --socratiq-brand-700: 3 56 41;
    --socratiq-brand-800: 2 42 31;
    --socratiq-brand-900: 2 29 22;
    --socratiq-brand-950: 1 17 13;
    --socratiq-muted: #4b5d57;
    --socratiq-border: #bfd6ce;
    --socratiq-hover: #f6fbf9;
    --socratiq-accent: #065f46;
    --socratiq-accent-strong: #044634;
    --socratiq-accent-rgb: 6, 95, 70;
    --socratiq-accent-bg: #dbf2ea;
    --socratiq-accent-hover: #b8e4d5;
}

:host([data-socratiq-theme="dark"]),
.socratiq-widget-root[data-socratiq-theme="dark"] {
    --socratiq-text: #e6edf3; /* Dark theme: light text */
    --socratiq-bg: #0d1117;
    --socratiq-muted: #9ca3af;
    --socratiq-border: #30363d;
    --socratiq-hover: #21262d;
    --socratiq-accent: #E85D75;
    --socratiq-accent-strong: #f07b8e;
    --socratiq-accent-rgb: 232, 93, 117;
    --socratiq-accent-bg: #2d161e;
    --socratiq-accent-hover: #3d1d28;
}

:host([data-socratiq-theme="dark"][data-socratiq-volume="vol2"]),
.socratiq-widget-root[data-socratiq-theme="dark"][data-socratiq-volume="vol2"] {
    --socratiq-accent: #6B9FD4;
    --socratiq-accent-strong: #8ab6e3;
    --socratiq-accent-rgb: 107, 159, 212;
    --socratiq-accent-bg: #152238;
    --socratiq-accent-hover: #1e304d;
}

:host([data-socratiq-theme="dark"][data-socratiq-volume="vol3"]),
.socratiq-widget-root[data-socratiq-theme="dark"][data-socratiq-volume="vol3"] {
    --socratiq-accent: #C084FC;
    --socratiq-accent-strong: #d2a4fd;
    --socratiq-accent-rgb: 192, 132, 252;
    --socratiq-accent-bg: #241436;
    --socratiq-accent-hover: #321c4a;
}

:host([data-socratiq-theme="dark"][data-socratiq-volume="vol4"]),
.socratiq-widget-root[data-socratiq-theme="dark"][data-socratiq-volume="vol4"] {
    --socratiq-accent: #34D399;
    --socratiq-accent-strong: #5eead4;
    --socratiq-accent-rgb: 52, 211, 153;
    --socratiq-accent-bg: #0d2922;
    --socratiq-accent-hover: #133b31;
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
    background-color: var(--socratiq-accent, #8e1537); /* Volume brand header background */
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
      background-color: var(--socratiq-accent, #8e1537); /* Tailwind class bg-blue-500 */
      color: white; /* Tailwind class text-white */
      padding: 0.5rem 1rem; /* Tailwind classes px-4 py-2 */
      border-radius: 0.25rem; /* Tailwind class rounded */
      transition: background-color 0.2s; /* Tailwind hover state */
      margin-bottom: 10px;
    }
    
    #submit-quiz-btn:hover {
      background-color: var(--socratiq-accent-strong, #74132f); /* Tailwind class hover:bg-blue-600 */
    }
    
    .submit-quiz-btn {
      margin-top: 1rem; /* Tailwind class mt-4 */
      width: 100%; /* Tailwind class w-full */
      background-color: var(--socratiq-accent, #8e1537); /* Tailwind class bg-blue-500 */
      color: white; /* Tailwind class text-white */
      padding: 0.5rem 1rem; /* Tailwind classes px-4 py-2 */
      border-radius: 0.25rem; /* Tailwind class rounded */
      transition: background-color 0.2s; /* Tailwind hover state */
      margin-bottom: 10px;
    }
    
    .submit-quiz-btn:hover {
      background-color: var(--socratiq-accent-strong, #74132f); /* Tailwind class hover:bg-blue-600 */
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
      color: var(--socratiq-text, #1f2328);
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
      height: 6px; /* Specified height */
      background: var(--socratiq-border, #d8c5cb);
      border-radius: 9999px;
      outline: none; /* Remove outline */
      opacity: 0.9;
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
      width: 18px; /* Width of the thumb */
      height: 18px; /* Height of the thumb */
      border-radius: 50%;
      background: var(--socratiq-accent, #8e1537);
      border: 2px solid var(--socratiq-bg, #ffffff);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      cursor: pointer; /* Cursor on hover */
    }
  
    /* Style for Mozilla Firefox */
    input[type=range]::-moz-range-thumb {
      width: 18px; /* Width of the thumb */
      height: 18px; /* Height of the thumb */
      border-radius: 50%;
      background: var(--socratiq-accent, #8e1537);
      border: 2px solid var(--socratiq-bg, #ffffff);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      cursor: pointer; /* Cursor on hover */
    }
  
    /* Style for the focus state */
    input[type=range]:focus {
      outline: none; /* Remove the outline */
    }
    
    /* Additional style for the focus state on Webkit browsers */
    input[type=range]:focus::-webkit-slider-thumb {
      background: var(--socratiq-accent, #74132f);
    }
  
    /* Additional style for the focus state on Mozilla Firefox */
    input[type=range]:focus::-moz-range-thumb {
      background: var(--socratiq-accent, #74132f);
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
    padding: 6px;
    width: min(480px, 94vw);
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
    background-color: var(--socratiq-bg, #ffffff);
    color: var(--socratiq-text, #1f2328);
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
  color: var(--socratiq-accent, #8e1537); /* Tailwind's blue-500 */
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
      --link-color: var(--socratiq-accent, #E85D75);
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
    --link-color: var(--socratiq-accent, #8e1537);
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
    color: var(--socratiq-accent, #8e1537);
  }
  
  
  #menu-toggle:checked~label {
    background-color: var(--socratiq-accent-strong, #74132f);
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
      box-shadow: inset 0 0 10px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.7), inset 0 0 20px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.5);
    }
    50% {
      box-shadow: inset 0 0 15px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 1), inset 0 0 30px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.8);
    }
    100% {
      box-shadow: inset 0 0 10px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.7), inset 0 0 20px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.5);
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
  .dropdown-item:hover { background: rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.08); color: var(--socratiq-accent, #8e1537); }
  .dropdown-item.active {
    background: rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.12);
    color: var(--socratiq-accent, #8e1537);
    font-weight: 600;
  }

  /* Draw-to-select active indicator on the ⋯ button */
  #more-options-button.draw-active {
    color: var(--socratiq-accent, #8e1537);
    background: rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.12);
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
    box-shadow: 0 0 0 2px rgba(var(--socratiq-accent-rgb, 142, 21, 55), 0.2);
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
  scrollbar-color: var(--socratiq-accent, #8e1537) transparent;
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
    border-top: 2px solid var(--socratiq-accent, #8e1537);
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
    border-left: 4px solid var(--socratiq-accent, #8e1537);
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
        background: var(--socratiq-accent, #8e1537);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
    }

    .boarding-button:hover {
        background: var(--socratiq-accent-strong, #74132f);
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
  border-color: var(--socratiq-accent, #8e1537);
}

.toggle-label {
  transition: background-color 0.2s ease;
}

.toggle-checkbox:checked + .toggle-label {
  background-color: var(--socratiq-accent, #8e1537);
}

/* Keep the existing deck controls usable when their modal narrows. */
/* Keep fixed light surfaces from overriding the widget's existing dark palette. */
:host([data-socratiq-theme="dark"]) .dark-mode {
  --text-color: var(--socratiq-text);
  --background-color: var(--socratiq-bg);
  --border-color: var(--socratiq-border);
  color: var(--socratiq-text);
}
:host([data-socratiq-theme="dark"]) #text-selection-menu > div,
:host([data-socratiq-theme="dark"]) .stats-modal .modal-content,
:host([data-socratiq-theme="dark"]) .stats-header,
:host([data-socratiq-theme="dark"]) .stats-footer {
  background-color: var(--socratiq-bg);
  color: var(--socratiq-text);
  border-color: var(--socratiq-border);
}
:host([data-socratiq-theme="dark"]) #markdown-preview,
:host([data-socratiq-theme="dark"]) .streamdown-wrapper {
  background-color: transparent;
  color: var(--socratiq-text);
}
:host([data-socratiq-theme="dark"]) #user-input {
  background-color: var(--socratiq-hover);
  border-color: var(--socratiq-border);
  color: var(--socratiq-text);
}
:host([data-socratiq-theme="dark"]) #user-input::placeholder { color: var(--socratiq-muted); }
:host([data-socratiq-theme="dark"]) #enter-btn,
:host([data-socratiq-theme="dark"]) #socratiq-study-close { color: var(--socratiq-text); }
:host([data-socratiq-theme="dark"]) #showStats,
:host([data-socratiq-theme="dark"]) #showViz { background-color: var(--socratiq-hover); }
:host([data-socratiq-theme="dark"]) #chapterList button[data-chapter][class~="bg-blue-800/50"] {
  color: var(--socratiq-text);
}
:host([data-socratiq-theme="dark"]) #text-selection-menu > div > .bg-zinc-100 {
  background-color: var(--socratiq-bg);
  color: var(--socratiq-muted);
}
:host([data-socratiq-theme="dark"]) .stats-body .bg-gray-50,
:host([data-socratiq-theme="dark"]) .stats-body .bg-white {
  background-color: var(--socratiq-hover);
  color: var(--socratiq-text);
}
:host([data-socratiq-theme="dark"]) .stats-body .text-gray-500,
:host([data-socratiq-theme="dark"]) .stats-body .text-gray-600,
:host([data-socratiq-theme="dark"]) .stats-body .text-gray-700 { color: var(--socratiq-muted); }
:host([data-socratiq-theme="dark"]) .stats-body .text-gray-800,
:host([data-socratiq-theme="dark"]) .stats-body .text-gray-900 { color: var(--socratiq-text); }
:host([data-socratiq-theme="dark"]) #socratiq-knowledge-graph-modal { color: #1f2328; }
:host([data-socratiq-theme="dark"]) .stats-body .bg-blue-50 { background-color: var(--socratiq-accent-bg); }
:host([data-socratiq-theme="dark"]) .stats-body .text-blue-600,
:host([data-socratiq-theme="dark"]) .stats-body .text-blue-700 { color: var(--socratiq-accent); }
/* Existing Chart.js labels use a dark ink color; keep their drawing surface light. */
:host([data-socratiq-theme="dark"]) .stats-body canvas { background-color: white; }
/* These sibling overlays must cover the workspace's z-9999 wrapper. */
.stats-modal, .verification-modal, #socratiq-knowledge-graph-modal, #helpModal {
  z-index: 100001 !important;
}
#sr-modal-content-container-controls { gap: 8px; flex-wrap: wrap; }
#sr-controls-container { display: flex; flex-wrap: wrap; gap: 8px; }
#sr-controls-container > button { margin-left: 0 !important; }
#showAddCard { align-self: flex-start; flex-shrink: 0; white-space: nowrap; }
@media (max-width: 640px) {
  /* The stacked deck keeps its original actions; duplicate floating actions
     would cover its search and import/export controls. */
  #sr-floating-controls { display: none !important; }
  .stats-modal .stats-layout { width: 100%; height: 100dvh; min-height: 0; padding: 0; }
  .stats-modal .stats-layout > span[aria-hidden] { display: none; }
  .stats-modal .modal-content { display: flex; flex-direction: column; width: 100%; height: 100dvh; max-height: 100dvh !important; margin: 0; padding: 0; overflow: hidden; }
  .stats-modal .stats-header { flex: 0 0 auto; padding: 16px; }
  .stats-modal .stats-header-row { position: relative; flex-direction: column; align-items: flex-start; gap: 10px; padding-right: 44px; }
  .stats-modal .stats-header h2 { font-size: 20px; line-height: 1.25; }
  .stats-modal .stats-header-actions { flex-wrap: wrap; gap: 8px; font-size: 12px; }
  .stats-modal .stats-header-actions > * { margin-left: 0 !important; }
  .stats-modal #badges-prompt { flex-basis: 100%; font-size: 12px; line-height: 1.4; }
  .stats-modal .close-modal { position: absolute; top: 0; right: 0; display: flex; align-items: center; justify-content: center; min-width: 44px; min-height: 44px; }
  .stats-modal .stats-body { flex: 1 1 auto; min-height: 0; max-height: none !important; padding: 16px; }
  .stats-modal .stats-footer { flex: 0 0 auto; padding: 12px; }
  #socratiq-knowledge-graph-modal #list-view { top: 60px !important; padding: 12px !important; }
  #socratiq-knowledge-graph-modal #list-view .list-item { flex-wrap: wrap; gap: 6px; padding: 8px 6px !important; }
  #socratiq-knowledge-graph-modal #list-view .list-item > div:first-child { flex: 1 1 100% !important; min-width: 0; }
  #socratiq-knowledge-graph-modal #list-view .list-item > div:first-child > span:last-child { overflow-wrap: anywhere; }
  #socratiq-knowledge-graph-modal > div {
    width: calc(100% - 24px) !important;
    height: calc(100dvh - 24px) !important;
    padding: 48px 12px 12px !important;
  }
  #socratiq-knowledge-graph-modal .knowledge-graph-layout { flex-direction: column; gap: 12px; min-height: 0; overflow-y: auto; }
  #socratiq-knowledge-graph-modal .knowledge-graph-viewport { flex: 0 0 400px !important; width: 100%; height: 400px !important; }
  #socratiq-knowledge-graph-modal .knowledge-graph-sidebar { flex: 0 0 auto; width: 100% !important; height: auto !important; padding: 16px !important; border-left: 0 !important; border-top: 1px solid #e2e8f0; }
  #socratiq-knowledge-graph-modal .quiz-items-container { flex: 0 0 auto; max-height: 180px; margin-bottom: 0 !important; }
  #socratiq-knowledge-graph-modal .knowledge-graph-actions { position: static !important; margin-top: 12px; padding-top: 12px !important; }
  #closeModal_sr, #sr-import-btn, #sr-download-btn, #copyButton, #sr-share-btn {
    min-width: 44px;
    min-height: 44px;
    align-items: center;
    justify-content: center;
  }
  #sr-modal-content_inner {
    top: 12px;
    width: calc(100% - 24px);
    max-height: calc(100dvh - 24px) !important;
    padding: 16px;
    margin-bottom: 12px;
  }
  #sr-modal-content_inner > .grid {
    display: flex;
    flex-direction: column;
    margin-top: 48px;
    overflow-y: auto !important;
  }
  #sr-sidebar {
    position: static !important;
    flex: 0 0 auto;
    max-height: none !important;
    overflow: visible !important;
    border-right: 0;
    border-bottom: 1px solid var(--socratiq-border);
    padding-right: 0;
    padding-bottom: 12px;
  }
  #sr-modal-content-container { flex: 0 0 auto; min-width: 0; max-height: none !important; padding-left: 0; overflow: visible; }
  #sr-modal-content-container > .flex.items-center.justify-between.w-full { flex-wrap: wrap; gap: 8px; }
  #sr-modal-content-container > .flex.items-center.justify-between.w-full > div:first-child { flex-basis: 100%; min-width: 0; }
  #sr-modal-content-container > .flex.items-center.justify-between.w-full > div:last-child { margin-left: 0; }
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

export const htmlContent = `
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SocratiQ</title>
  <!-- Ensure the Tailwind CSS CDN is linked in the <head> -->
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@latest/dist/tailwind.min.css" rel="stylesheet">
</head>

<body id="mybody" class="min-h-screen z-9999">

  <!-- Overlay container with fixed positioning -->
  <div class="fixed inset-0 z-9999" style="pointer-events: none;">
    <!-- Slide-out menu container -->
    <div id="text-selection-menu"
      class="absolute right-0 top-0 h-full transform translate-x-full transition-transform duration-300 ease-in-out"
      style="width: 100%; max-width: 400px; pointer-events: auto;">
      <!-- Actual slide-out menu with defined max-width -->
      <div class="p-4 bg-gray-100 dark:bg-zinc-700 shadow-lg rounded-lg h-full flex flex-col overflow-auto"
        style="max-width: 400px; box-shadow: 4px 0 15px rgba(0,0,0,0.25);">

        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg text-zinc-800 dark:text-white">
            SocratiQ
          </h2>
          <div class="flex space-x-2">

            <span id="new-chat-btn" 
              class="hover:text-blue-300 text-sm text-zinc-600 dark:text-zinc-400">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                stroke="currentColor" class="w-6 h-6" style="width: 1.5rem; height: 1.5rem;">
                <path stroke-linecap="round" stroke-linejoin="round"
                  d="M2.25 12.76c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.076-4.076a1.526 1.526 0 0 1 1.037-.443 48.282 48.282 0 0 0 5.68-.494c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z" />
              </svg>


            </span>


            <!-- <span id="spaced-repetition-btn"
            class="hover:text-blue-300 text-sm text-zinc-600 dark:text-zinc-400">
          <svg data-slot="icon" fill="none" stroke-width="1.5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"
           stroke="currentColor" class="w-6 h-6" style="width: 1.5rem; height: 1.5rem;">
<path stroke-linecap="round" stroke-linejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z"></path>
</svg>

          </span> -->

  <span id="knowledge-graph-btn" 
              class="hover:text-blue-300 text-sm text-zinc-600 dark:text-zinc-400">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"class="w-6 h-6" style="width: 1.5rem; height: 1.5rem;">
  <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23-.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
</svg>


            </span>


            <span id="chat-quiz-btn"
              class="hover:text-blue-300 text-sm text-zinc-600 dark:text-zinc-400">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                stroke="currentColor" class="w-6 h-6" style="width: 1.5rem; height: 1.5rem;">
                <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6a7.5 7.5 0 1 0 7.5 7.5h-7.5V6Z" />
                <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0 0 13.5 3v7.5Z" />
              </svg>

            </span>

            <span id="help-btn"
              class="hover:text-blue-300 text-sm text-zinc-600 dark:text-zinc-400">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                stroke="currentColor" class="w-6 h-6" style="width: 1.5rem; height: 1.5rem;">
                <path stroke-linecap="round" stroke-linejoin="round"
                  d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z" />
              </svg>

            </span>

            <span id="settings-btn" 
              class="hover:text-blue-300 text-sm text-zinc-600 dark:text-zinc-400">


              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                stroke="currentColor" class="w-6 h-6" style="width: 1.5rem; height: 1.5rem;">
                <path stroke-linecap="round" stroke-linejoin="round"
                  d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75" />
              </svg>
              <!-- </a> -->
            </span>
          </div>
        </div>
        <div id="message-container" class="overflow-y-auto space-y-6 flex-grow" style="width: fit-content">


          <div id='human-message_2'
            class="p-3 ml-10 max-w-xs bg-blue-500 dark:bg-zinc-600 rounded-lg mb-2 chat-message">
            <p class="text-sm text-white dark:text-zinc-200">
              What is SocratiQ?
            </p>
          </div>

          <div id="ai-message_2"
            class="p-3 relative max-w-xs bg-white dark:bg-zinc-600 space-y-2 rounded-lg mb-2 chat-message" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">
<!--   
                        <div class="w-full flex justify-end">
                        <div class="difficulty-dropdown inline-flex items-center text-xs bg-gray-100 dark:bg-zinc-700 rounded-md px-2 py-1 mb-2">
                          <div class="relative">
                            <button class="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white">
                              <span class="current-difficulty-level">🚗 Intermediate</span>
                              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                              </svg>
                            </button>
                            <div class="difficulty-options hidden absolute top-full left-0 mt-1 w-40 bg-white dark:bg-zinc-800 rounded-md shadow-lg py-1 z-50" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">
                              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🚲 Beginner</div>
                              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🚗 Intermediate</div>
                              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🚁 Advanced</div>
                              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🛸 AGI</div>
                            </div>
                          </div>
                        </div>
                      </div> -->
            <div id="markdown-preview" class="text-sm text-zinc-800 dark:text-zinc-200 chat-message">
              SocratiQ is an AI Generative Learning Assistant, designed to make learning more efficient, engaging and
              accessible. We welcome your <a href="https://forms.gle/jmWJcdzN2TXXjHWn7" target="_blank"
                class="text-blue-500 hover:text-blue-700" title="Provide your feedback here">Feedback</a>. Learn more
              about <a href="https://www.youtube.com/watch?v=mIT9nIsxCe0" target="_blank"
                class="text-blue-500 hover:text-blue-700"
                title="Watch the SocratiQ Introduction on YouTube">SocratiQ</a>.
            </div>

            <div id="utility-btn-container" class="absolute bottom-1 right-2 text-blue-400 flex space-x-2 p-2">
              <!-- Buttons with icons -->

              <button id='highlight-btn' class="w-4 h-4  flex items-center justify-center hover:text-blue-700"
                title="Highlight">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                  stroke="currentColor" class="icons">
                  <path stroke-linecap="round" stroke-linejoin="round"
                    d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
                </svg>
              </button>

                   <button id='sr-send-btn' class="w-4 h-4  mr-2  flex items-center justify-center hover:text-blue-700"
               >
                

                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="icons">
  <path stroke-linecap="round" stroke-linejoin="round" 
  d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z" />
</svg>

              </button>

              <button id='share-btn' class="w-4 h-4  flex items-center justify-center hover:text-blue-700"
                title="Share"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                  stroke="currentColor" class="icons">
                  <path stroke-linecap="round" stroke-linejoin="round"
                    d="M7.217 10.907a2.25 2.25 0 1 0 0 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186 9.566-5.314m-9.566 7.5 9.566 5.314m0 0a2.25 2.25 0 1 0 3.935 2.186 2.25 2.25 0 0 0-3.935-2.186Zm0-12.814a2.25 2.25 0 1 0 3.933-2.185 2.25 2.25 0 0 0-3.933 2.185Z" />
                </svg>
              </button>
              <button id='copy-btn' class="w-4 h-4  mr-1 mt-1  flex items-center justify-center hover:text-blue-700"
                title="Copy"><svg id="copy-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                  stroke-width="1.5" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round"
                    d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75" />
                </svg>
              </button>
              <button id='download-btn' class="w-4 h-4  mr-2  flex items-center justify-center hover:text-blue-700"
                title="Download"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                  stroke-width="1.5" stroke="currentColor" class="icons">
                  <path stroke-linecap="round" stroke-linejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
              </button>

             

            </div>


            <div id="reference-btn-container_temp" class="flex space-x-2 text-xs font-mono">
              <a href="https://harvard-edge.github.io/cs249r_book/"
                class="inline-block bg-blue-100 text-zinc-900 px-4 py-0.5 rounded-md" target="_blank"
                rel="noopener noreferrer" title="TinyML Textbook">1</a>
              <!-- Omitted buttons as they are not links -->
              <a href="https://github.com/harvard-edge/cs249r_book"
                class="inline-block bg-blue-100 text-zinc-900 px-4 py-0.5 rounded-md" target="_blank"
                rel="noopener noreferrer" title="TinyML GitHub Repo">2</a>
              <a href="https://www.edx.org/certificates/professional-certificate/harvardx-tiny-machine-learning"
                class="inline-block bg-blue-100 text-zinc-900 px-4 py-0.5 rounded-md" target="_blank"
                rel="noopener noreferrer" title="TinyML on EdX">3</a>

            </div>




          </div>




          <!-- Additional chat messages will go here -->
          <div id="scroll-target"></div> <!-- Dummy div at the bottom -->

        </div>

        <div>

        </div>

        <div id="input-container" class="mt-4 relative">
          <!-- <div id="ai-prompt-label" style="font-size: 12px; color: rgb(85, 85, 85); margin-bottom: 8px; display: block;">Included context:</div> -->

          <div style="display: flex; gap: 8px; margin-bottom: 8px;">
            <button id="context-button" title="Context">
              <div style="display: flex; align-items: center;"> 
                <span style="font-size: 1.2rem; font-weight: bold; color: #6b7280;">@</span>
                <span style="font-size: 12px; margin-left: 4px;">Add Context</span>
              </div>
            </button>

            <button id="quiz-button" title="Quiz">
              <div style="display: flex;"> 
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 1.2rem; height: 1.2rem;">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M8.242 5.992h12m-12 6.003H20.24m-12 5.999h12M4.117 7.495v-3.75H2.99m1.125 3.75H2.99m1.125 0H5.24m-1.92 2.577a1.125 1.125 0 1 1 1.591 1.59l-1.83 1.83h2.16M2.99 15.745h1.125a1.125 1.125 0 0 1 0 2.25H3.74m0-.002h.375a1.125 1.125 0 0 1 0 2.25H2.99" />
                </svg>
                <span style="font-size: 12px; margin-left: 4px;">Quiz me</span>
              </div>
            </button>

            <button id="flashcards-button" title="Flashcards">
              <div style="display: flex;"> 
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 1.2rem; height: 1.2rem;">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z" />
                </svg>
                <span style="font-size: 12px; margin-left: 4px;">Flashcards</span>
              </div>
            </button>

            <div class="relative flex items-center justify-center">
              <button id="more-options-button" title="More Options">
                <div style="display: flex; align-items: center;"> 
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width: 1.2rem; height: 1.2rem;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M6.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM12.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM18.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Z" />
                  </svg>
                </div>
              </button>
              
              <!-- Dropdown Menu -->
              <div id="more-options-dropdown" class="more-options-dropdown hidden">
                <div class="dropdown-content">
                  <button id="draw-select-btn" class="dropdown-item">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width:1rem;height:1rem;flex-shrink:0;">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487zm0 0L19.5 7.125" />
                    </svg>
                    <span>Draw to Select</span>
                  </button>
                  <button id="meditation-btn" class="dropdown-item">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width:1rem;height:1rem;flex-shrink:0;">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386-1.591 1.591M21 12h-2.25m-.386 6.364-1.591-1.591M12 18.75V21m-4.773-4.227-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0Z" />
                    </svg>
                    <span>Meditation Timer</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
          <!-- Copy container with dynamic height -->
          <div id="copy-container" class="hidden">


          </div>

          <!-- Input jj field with embedded button -->
          <div class="input-wrapper relative">
            <input type="text" id="user-input"
              class="w-full pl-2 pr-10 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-zinc-600 dark:border-zinc-500 dark:text-white"
              placeholder="type '@' to reference a section..." />
            <button id="enter-btn" class="absolute inset-y-0 right-0 flex items-center pr-3 hover:text-blue-500"
              type="submit">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
                stroke="currentColor" class="w-6 h-6" style="transform: rotate(-45deg);">
                <path stroke-linecap="round" stroke-linejoin="round"
                  d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
              </svg>
            </button>
          </div>
        </div>



        <div class="bg-zinc-100 text-gray-500 p-1 dark:bg-zinc-200/50" style="font-size: 8pt">
          <!-- <p class="font-bold">Be cautious</p> -->
          <p>Information provided here may not always be accurate.<span> <a href="#" id="openModal-feedback"
                class="underline text-blue-500 hover:text-zinc-800 dark:hover:text-zinc-900">Provide feedback</a>
            </span></p>
        </div>
      </div>

    </div>


    <label for="menu-toggle" id="toggleButton"
      class="cursor-pointer p-4 bg-blue-500 text-white absolute top-1/2 right-0 transform -translate-y-1/2"
      style="pointer-events: auto; transform: translateY(-50%) translateX(-5px); z-index:9999">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
        class="w-6 h-6">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M7.5 21 3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
      </svg>
    </label>

  </div>

  <!-- Hidden checkbox to hold the state -->
  <input type="checkbox" id="menu-toggle" class="hidden" />

  <style>
    .no-scrollbar::-webkit-scrollbar {
      display: none;
    }
    
    /* Modal styles */
    .modal {
      display: none;
      position: fixed;
      z-index: 1000;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.5);
      backdrop-filter: blur(2px);
    }
    
    .modal.show {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .modal > div {
      position: relative;
      z-index: 1001;
    }
    
    /* Dropdown styles */
    .more-options-dropdown {
      position: absolute;
      top: 100%;
      right: 0;
      z-index: 50;
      margin-top: 0.25rem;
    }
    
    .dropdown-content {
      background-color: white;
      border: 1px solid #e5e7eb;
      border-radius: 0.375rem;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
      min-width: 160px;
      padding: 0.25rem 0;
    }
    
    .dark .dropdown-content {
      background-color: #1f2937;
      border-color: #374151;
    }
    
    .dropdown-item {
      display: flex;
      align-items: center;
      width: 100%;
      padding: 0.5rem 0.75rem;
      text-align: left;
      font-size: 0.875rem;
      color: #374151;
      background: none;
      border: none;
      cursor: pointer;
      transition: background-color 0.15s ease-in-out;
    }
    
    .dropdown-item:hover {
      background-color: #f3f4f6;
    }
    
    .dark .dropdown-item {
      color: #d1d5db;
    }
    
    .dark .dropdown-item:hover {
      background-color: #374151;
    }
    
    /* Ensure the more options button aligns with other buttons */
    #more-options-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.5rem;
      border: none;
      border-radius: 0.375rem;
      background-color: transparent;
      color: #374151;
      font-size: 0.875rem;
      transition: all 0.15s ease-in-out;
    }
    
    #more-options-button:hover {
      background-color: #f3f4f6;
    }
    
    .dark #more-options-button {
      background-color: transparent;
      color: #d1d5db;
    }
    
    .dark #more-options-button:hover {
      background-color: #374151;
    }
    
    /* Difficulty dropdown styles */
    .difficulty-content-collapsed {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.3s ease-in-out;
    }
    
    .difficulty-content-expanded {
      max-height: 500px;
      overflow: hidden;
      transition: max-height 0.3s ease-in-out;
    }
    
    #difficulty-dropdown-toggle.rotated {
      transform: rotate(180deg);
    }
    
    /* Toggle switch styles */
    .toggle-checkbox:checked {
      right: 0;
      border-color: #3b82f6;
    }
    .toggle-checkbox:checked + .toggle-label {
      background-color: #3b82f6;
    }
  </style>
  </div>
  </div>
  <div id="success-notice" style="
  display: none;
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: #f8f9fa;
  padding: 10px 15px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1),
              0 0 15px rgba(66, 153, 225, 0.3);
  font-family: system-ui, -apple-system, sans-serif;
  color: #2d3748;
  animation: glow 2s infinite alternate;
">
    <span style="margin-right: 8px">Press</span>
    <kbd style="
      background: #fff;
      border: 1px solid #d1d5db;
      border-radius: 3px;
      padding: 2px 6px;
      box-shadow: 0 1px 1px rgba(0,0,0,0.2);
      font-family: monospace;
  " id="shortcut-modifier-key">${/Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent) ? '⌘' : 'Ctrl'}</kbd>
    <span style="margin: 0 4px">+</span>
    <kbd style="
      background: #fff;
      border: 1px solid #d1d5db;
      border-radius: 3px;
      padding: 2px 6px;
      box-shadow: 0 1px 1px rgba(0,0,0,0.2);
      font-family: monospace;
  ">/</kbd> to open chat
    <style>
      @keyframes glow {
        from {
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1),
            0 0 15px rgba(66, 153, 225, 0.3);
        }

        to {
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1),
            0 0 20px rgba(66, 153, 225, 0.5);
        }
      }
    </style>
  </div>

  <div id="error-notice" style="
      display: none;
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: lightcoral;
      padding: 10px;
      border-radius: 5px;
    ">
    Disconnected
  </div>

  <div id="load_chats"></div>
  <div id="modal1" class="modal">
    <!-- Modal content -->

    <div class="bg-white p-6 rounded-lg shadow-lg max-w-sm mx-auto mt-10" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">
      <div class="flex justify-between items-center mb-4">
        <span class="text-lg font-semibold dark:text-white">Settings</span>
        <button id='close-btn' class="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5"
            stroke="currentColor" class="w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>

      <!-- Use your own AI Section -->
      <div class="mb-6 p-4 border border-zinc-200 dark:border-zinc-700 rounded-lg" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-md font-semibold text-zinc-900 dark:text-white">Use your own AI</h3>
          
          <!-- Custom API Toggle Switch -->
          <div class="flex items-center space-x-3">
            <span class="text-sm text-zinc-600 dark:text-zinc-400">Use Custom API</span>
            <div class="relative inline-block w-12 align-middle select-none">
              <input type="checkbox" id="custom-api-toggle" 
                class="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"/>
              <label for="custom-api-toggle" 
                class="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"></label>
            </div>
          </div>
        </div>
        
        <!-- Custom API Configuration (initially hidden) -->
        <div id="custom-api-config" class="hidden">
          <!-- API Provider Dropdown -->
        <div class="mb-4">
          <label for="ai-provider" class="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">AI Provider</label>
          <select id="ai-provider" class="w-full px-3 py-2 border border-zinc-300 dark:border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-zinc-700 dark:text-white">
            <option value="">Select an AI provider</option>
            <option value="google-gemini">Google Gemini API</option>
            <option value="open-router">Open Router API</option>
            <option value="groq">GROQ API</option>
            <option value="ollama">OLLAMA API</option>
          </select>
        </div>

        <!-- Model Input -->
        <div class="mb-4">
          <label for="api-model" class="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">Model</label>
          <input type="text" id="api-model" 
            class="w-full px-3 py-2 border border-zinc-300 dark:border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-zinc-700 dark:text-white"
            placeholder="Model name (e.g., gemini-2.5-flash, gpt-3.5-turbo)">
        </div>

        <!-- API Endpoint URL Input -->
        <div class="mb-4">
          <label for="api-endpoint" class="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">API Endpoint URL</label>
          <input type="url" id="api-endpoint" 
            class="w-full px-3 py-2 border border-zinc-300 dark:border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-zinc-700 dark:text-white"
            placeholder="https://api.example.com/v1/chat/completions">
        </div>

        <!-- API Key Input -->
        <div class="mb-4">
          <label for="api-key" class="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">API Key (Optional)</label>
          <input type="password" id="api-key" 
            class="w-full px-3 py-2 border border-zinc-300 dark:border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-zinc-700 dark:text-white"
            placeholder="Enter your API key (leave empty for local APIs like Ollama)">
        </div>

        <!-- Save Locally Checkbox -->
        <div class="mb-2">
          <label class="inline-flex items-center">
            <input type="checkbox" id="save-locally" 
              class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-zinc-300 dark:border-zinc-600 rounded">
            <span class="ml-2 text-sm text-zinc-700 dark:text-zinc-300">Save locally</span>
          </label>
        </div>
        
        <!-- Privacy Warning -->
        <div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
          <p class="text-xs text-yellow-800 dark:text-yellow-200">
            <strong>Privacy Notice:</strong> Only use the save locally feature on private computers. Your API credentials will be stored in your browser's local storage.
          </p>
        </div>
        
        <!-- Reset Button -->
        <div class="mt-4 pt-4 border-t border-zinc-200 dark:border-zinc-700">
          <button id="reset-custom-api" 
            class="w-full bg-red-50 hover:bg-red-100 text-red-600 font-medium py-2 px-4 rounded-md focus:outline-none border border-red-200 transition-colors">
            Reset to Default Settings
          </button>
        </div>
        </div>
        
        <!-- Default API Info (shown when custom API is disabled) -->
        <div id="default-api-info" class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
          <p class="text-sm text-blue-800 dark:text-blue-200">
            <strong>Using SocratiQ AI:</strong> You're currently using our default AI providers (GROQ, Gemini, etc.) with automatic fallback and optimization.
          </p>
        </div>
      </div>

      <!-- Add this at the top of the settings modal -->
      <div class="mb-6 border-b pb-4">
        <label class="flex items-center justify-between">
          <span class="text-sm font-medium text-zinc-900 dark:text-zinc-300">
            Push content when menu opens
          </span>
          <div class="relative inline-block w-10 mr-2 align-middle select-none">
            <input type="checkbox" id="push-content-toggle" 
              class="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"/>
            <label for="push-content-toggle" 
              class="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"></label>
          </div>
        </label>
      </div>

      <!-- Difficulty Level Dropdown -->
      <div class="mb-6">
        <button id="difficulty-dropdown-toggle" class="w-full flex justify-between items-center p-3 bg-gray-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg hover:bg-gray-100 dark:hover:bg-zinc-700 transition-colors">
          <span class="text-sm font-medium text-zinc-900 dark:text-white">Understanding Level Info</span>
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 text-zinc-500 dark:text-zinc-400 transition-transform duration-200">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
          </svg>
        </button>
        
        <div id="difficulty-content" class="difficulty-content-collapsed mt-2">
          <div class="p-4 border border-zinc-200 dark:border-zinc-700 rounded-lg">
            <div class="grid grid-cols-2 gap-4">
              <div class="difficulty-level beginner-level">
                <div class="flex items-center gap-2">
                  <span class="text-xl">🚲</span>
                  <span class="font-semibold text-sm dark:text-white">Beginner</span>
                </div>
                <p class="text-xs text-zinc-600 dark:text-zinc-400">For learners who are new to machine learning systems and are building foundational knowledge of concepts, tools, and basic implementations.</p>
              </div>
              
              <div class="difficulty-level intermediate-level">
                <div class="flex items-center gap-2">
                  <span class="text-xl">🚗</span>
                  <span class="font-semibold text-sm dark:text-white">Intermediate</span>
                </div>
                <p class="text-xs text-zinc-600 dark:text-zinc-400">For learners who have a working understanding of machine learning principles and are ready to design and optimize systems for real-world applications.</p>
              </div>
              
              <div class="difficulty-level advanced-level">
                <div class="flex items-center gap-2">
                  <span class="text-xl">🚁</span>
                  <span class="font-semibold text-sm dark:text-white">Advanced</span>
                </div>
                <p class="text-xs text-zinc-600 dark:text-zinc-400">For learners with significant experience in machine learning systems, focused on tackling complex problems, scaling solutions, and innovating in the field.</p>
              </div>
              
              <div class="difficulty-level agi-level">
                <div class="flex items-center gap-2">
                  <span class="text-xl">🛸</span>
                  <span class="font-semibold text-sm dark:text-white">Bloom's Taxonomy</span>
                </div>
                <p class="text-xs text-zinc-600 dark:text-zinc-400">Bloom's Taxonomy: Bloom's Taxonomy is an educational framework classifying cognitive skills from basic recall to complex evaluation. 
                  <a class="text-blue-500 text-sm hover:text-zinc-800 dark:hover:text-zinc-900"  href="https://en.wikipedia.org/wiki/Bloom%27s_taxonomy" target="_blank">Read More</a></p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="mb-6">
        <label for="understanding-level" class="block mb-2 font-medium text-zinc-900 dark:text-zinc-300">Understanding
          Level</label>
        <input type="range" id="understanding-slider" name="understanding-level" min="0" max="3" value="2"
          class="w-full h-2 bg-zinc-200 rounded-lg appearance-none cursor-pointer dark:bg-zinc-700" />
        <div class="flex justify-between text-xs text-zinc-500 dark:text-zinc-400 mt-1">
          <span id="low-understanding">🚲</span>
          <span id="medium-understanding">🚗</span>
          <span id="high-understanding">🚁</span>
          <span id="super-high-understanding">🛸</span>

        </div>
      </div>

      <div class="hidden">
        <div class="mb-4">
          <label for="choices">Choose an LLM:</label>
          <select id="choices" name="choices">
            <option value="gemma-7b-It">Gemma-7b-it</option>
            <option value="llama3-70b-8192">Llama3-70b-8192</option>
            <option value="llama3-8b-8192" selected>Llama3-8b-8192</option>
            <option value="mixtral-8x7b-32768">Mixtral-8x7b-32768</option>
          </select>
        </div>

        <div class="mb-4">

          <label class="inline-flex items-center">
            <input id="show-answers" type="checkbox"
              class="text-blue-600 form-checkbox rounded border-zinc-300 dark:border-zinc-700 dark:bg-zinc-800 dark:checked:bg-blue-600"
              checked />
            <span class="ml-2 text-zinc-900 dark:text-zinc-300">Show answers</span>
          </label>
        </div>



        <div class="mb-4">

          <label class="inline-flex items-center">
            <input id="Enable-highlight-menu" type="checkbox"
              class="text-blue-600 form-checkbox rounded border-zinc-300 dark:border-zinc-700 dark:bg-zinc-800 dark:checked:bg-blue-600"
              checked />
            <span class="ml-2 text-zinc-900 dark:text-zinc-300">Enable Highlight Menu</span>
          </label>
        </div>


        <div class="mb-6">
          <label class="inline-flex items-center">
            <input id="show-chain-of-thought" type="checkbox"
              class="text-blue-600 form-checkbox rounded border-zinc-300 dark:border-zinc-700 dark:bg-zinc-800 dark:checked:bg-blue-600"
              checked />
            <span class="ml-2 text-zinc-900 dark:text-zinc-300">Show chain of thought</span>
          </label>
        </div>
      </div>
      <!-- <button
        class="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline">
        Save
      </button> -->
    </div>

    <!-- <div class="mb-4">
      <label class="inline-flex items-center">
        <input id="Apply-blooms-taxonomy" type="checkbox"
          class="text-blue-600 form-checkbox rounded border-zinc-300 dark:border-zinc-700 dark:bg-zinc-800 dark:checked:bg-blue-600"
          checked />
        <span class="ml-2 text-zinc-900 dark:text-zinc-300">Apply Bloom's Taxonomy</span>
      </label>
    </div> -->

  </div>


  <div id="modal_feedback">

  </div>

  <!-- Quiz Options Modal -->
  <div id="quizModal" style="
    display: none;
    position: fixed;
    inset: 0;
    z-index: 99999;
    background: rgba(0,0,0,0.45);
    backdrop-filter: blur(2px);
    align-items: center;
    justify-content: center;
  ">
    <div style="
      background: var(--socratiq-bg, #ffffff);
      color: var(--socratiq-text, #1f2328);
      border-radius: 16px;
      padding: 28px 24px 24px;
      width: min(420px, 92vw);
      box-shadow: 0 20px 60px rgba(0,0,0,0.25);
      position: relative;
      animation: quizModalIn 0.18s ease;
    ">
      <style>
        @keyframes quizModalIn {
          from { opacity: 0; transform: scale(0.95) translateY(8px); }
          to   { opacity: 1; transform: scale(1) translateY(0); }
        }
        #quizModal .quiz-option-card {
          display: flex;
          align-items: flex-start;
          gap: 14px;
          padding: 16px;
          border-radius: 12px;
          border: 1.5px solid var(--socratiq-border, #e5e7eb);
          cursor: pointer;
          transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
          background: var(--socratiq-bg, #ffffff);
          text-align: left;
          width: 100%;
          margin-bottom: 12px;
        }
        #quizModal .quiz-option-card:hover {
          border-color: #6366f1;
          box-shadow: 0 0 0 3px rgba(99,102,241,0.12);
          background: rgba(99,102,241,0.04);
        }
        #quizModal .quiz-option-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }
        #quizModal .quiz-option-title {
          font-size: 0.95rem;
          font-weight: 600;
          margin-bottom: 3px;
          color: var(--socratiq-text, #1f2328);
        }
        #quizModal .quiz-option-desc {
          font-size: 0.8rem;
          color: var(--socratiq-text-muted, #6b7280);
          line-height: 1.4;
        }
        #quizModal .quiz-modal-close {
          position: absolute;
          top: 14px;
          right: 16px;
          background: none;
          border: none;
          cursor: pointer;
          color: var(--socratiq-text-muted, #9ca3af);
          padding: 4px;
          border-radius: 6px;
          line-height: 1;
          font-size: 1.1rem;
        }
        #quizModal .quiz-modal-close:hover { color: var(--socratiq-text, #1f2328); }
      </style>

      <button class="quiz-modal-close" id="close-quiz-modal" aria-label="Close">✕</button>

      <div style="margin-bottom: 20px;">
        <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; color: var(--socratiq-text, #1f2328);">Choose Quiz Type</div>
        <div style="font-size: 0.8rem; color: var(--socratiq-text-muted, #6b7280);">Select how you'd like to be tested</div>
      </div>

      <!-- Section Quiz -->
      <button class="quiz-option-card" id="section-quiz-btn">
        <div class="quiz-option-icon" style="background: rgba(99,102,241,0.1);">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="#6366f1">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25Z" />
          </svg>
        </div>
        <div>
          <div class="quiz-option-title">Section Quiz</div>
          <div class="quiz-option-desc">Test your understanding of the current section with focused questions</div>
        </div>
      </button>

      <!-- Cumulative Quiz -->
      <button class="quiz-option-card" id="cumulative-quiz-btn" style="margin-bottom: 0;">
        <div class="quiz-option-icon" style="background: rgba(16,185,129,0.1);">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="#10b981">
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3v11.25A2.25 2.25 0 0 0 6 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0 1 18 16.5h-2.25m-7.5 0h7.5m-7.5 0-1 3m8.5-3 1 3m0 0 .5 1.5m-.5-1.5h-9.5m0 0-.5 1.5m.75-9 3-3 2.148 2.148A12.061 12.061 0 0 1 16.5 7.605" />
          </svg>
        </div>
        <div>
          <div class="quiz-option-title">Cumulative Quiz</div>
          <div class="quiz-option-desc">Review material from multiple chapters to reinforce long-term learning</div>
        </div>
      </button>
    </div>
  </div>

  <div id="helpModal" class="hidden">

  </div>

  <!-- Cumulative quiz button removed -->
      </div>
    </div>
  </div>

  <div class="hidden" id="bag-of-stuff">

    <div id='human-message'
      class="p-3 text-white ml-10 max-w-xs bg-blue-500 dark:bg-zinc-600 rounded-lg mb-2 human-message-chat">
      <p class="text-sm text-white dark:text-zinc-200">
        An error occurred. Please try again.
      </p>
    </div>

    <div id="ai-message" class="p-3 max-w-xs bg-white dark:bg-zinc-600 space-y-2 rounded-lg mb-2 ai-message-chat" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">

      <div class="w-full flex justify-end">
        <div class="difficulty-dropdown inline-flex items-center text-xs bg-gray-100 dark:bg-zinc-700 rounded-md px-2 py-1 mb-2">
          <div class="relative">
            <button class="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white">
              <span class="current-difficulty-level">🚗 Intermediate</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
              </svg>
            </button>
            <div class="difficulty-options hidden absolute top-full left-0 mt-1 w-40 bg-white dark:bg-zinc-800 rounded-md shadow-lg py-1 z-50" style="background-color: var(--socratiq-bg, #ffffff); color: var(--socratiq-text, #1f2328);">
              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🚲 Beginner</div>
              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🚗 Intermediate</div>
              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🚁 Advanced</div>
              <div class="difficulty-option px-4 py-2 hover:bg-gray-100 dark:hover:bg-zinc-700 cursor-pointer">🛸 Bloom's Taxonomy</div>
            </div>
          </div>
        </div>
      </div>
      <div id="progress"></div>
      <div id="markdown-preview" class="text-sm text-zinc-800 dark:text-zinc-200 markdown-preview-container">
        An error occurred. Please try again.
      </div>

    </div>













    <div id="reference-btn-container" class="flex space-x-2 text-xs font-mono">
      <!-- <button id="reference-btn" class="bg-blue-100 text-zinc-900 px-4 py-0.5 rounded-md">1</button> -->

    </div>

    <a href="https://example.com" id="reference-btn"
      class="inline-block bg-blue-100 text-zinc-900 px-4 py-0.5 rounded-md">1</a>



    <!-- The Modal -->

  </div>

<!-- tooltip -->

<!-- Add this near the end of your body tag -->
<!-- <div id="tooltip-template" class="tooltip-container" style="display: none;">
  <div class="tooltip-content"></div>
  <div class="tooltip-arrow"></div>
</div> -->
</body>

</html>
`

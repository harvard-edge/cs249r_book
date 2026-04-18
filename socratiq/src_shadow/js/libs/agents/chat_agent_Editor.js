// import { TextSelection } from 'prosemirror-state';
import {insertTextAndMoveCursor, appendTextWithNewTextNode, appendTextToNode, getCurrentNode} from '../text_editor/insert_into_editor.js';
const url = "http://localhost:3000/query-memory-agent"; // Replace with your actual endpoint URL
const url_no_memory = "http://localhost:3000/query-agent";
const url_no_memory_stream = "http://localhost:3000/query-agent_stream"
const url_no_memory_stream_tiny = "http://localhost:3000/query_agent_stream_tiny"







// function insertTextAndMoveCursor(text, editor) {
//   console.log(editor)
//   console.log("i am editor.view", editor.view)
//   const { state, dispatch } = editor.view;
//   const { tr, selection } = state;
//   const insertPos = selection.from; // Get current cursor position
//   const endPos = insertPos + text.length; // Calculate the end position after insertion

//   // Step 1: Insert the text at the current cursor position
//   const transaction = tr.insertText(text, insertPos);

//   // Step 2: Move the cursor to the end of the inserted text
//   // Create a new TextSelection that points to the end position
//   const newSelection = TextSelection.create(transaction.doc, endPos);
//   const transactionWithNewCursor = transaction.setSelection(newSelection);

//   // Dispatch the transaction to update the editor's state
//   dispatch(transactionWithNewCursor);
// // 
//   // Optionally, focus the editor after updating
//   editor.view.focus();
// }


export async function query_agent(query, token, editor) {
    let result = '';

    const currentNode = getCurrentNode(editor)

    // TRACE: Track which AI endpoint is being called
    console.trace(`[AI_TRACE] chat_agent_Editor.js - Calling AI endpoint: ${url_no_memory_stream_tiny}`, {
      queryPreview: query?.prompt?.substring(0, 100) + '...' || 'No prompt',
      editorType: editor?.constructor?.name || 'Unknown'
    });

    try {
      const response = await fetch(url_no_memory_stream_tiny, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(query),
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
  
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk =  decoder.decode(value, { stream: true });
        result += chunk;
        // appendTextToNode(editor,currentNode,chunk)
        appendTextWithNewTextNode(chunk,editor)

        // result += decoder.decode(value, { stream: true });
      }
      
      return result;
    } catch (error) {
      console.error("Failed to query memory agent:", error);
      throw error; // It's generally a good idea to rethrow the error after logging it.
    }
  }
  
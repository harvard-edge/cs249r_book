// import Worker from './worker.js';
import { CustomTextSplitter } from "../vector_db/vectordb/splitters/text_splitter";
const worker_base = new Worker(new URL('./worker.js', import.meta.url));

const callbacks = new Map(); // To track callbacks associated with each action
let messageId = 0; // A simple counter to ensure uniqueness

worker_base.onmessage = function(event) {
    const callback = callbacks.get(event.data.id); // Retrieve callback by message ID

    if (event.data.status === 'success' && callback) {

        callback.resolve(event.data.data);
    } else if (callback) {
        callback.reject(event.data.message);
    }
    callbacks.delete(event.data.id); // Clean up the callback once it's called
};

const intervalSize = 200; // Example interval size
const overlapSize = 20; // Example overlap size
const textSplitter = new CustomTextSplitter(intervalSize, overlapSize, "\n\n");

const numCores = navigator.hardwareConcurrency || 2;
const workers = [];
for (let i = 0; i < numCores; i++) {
  const worker = new Worker(new URL("./worker.js", import.meta.url));
  worker.onmessage = handleWorkerMessage;
  workers.push(worker);
}



function handleWorkerMessage(event) {
  const callback = callbacks.get(event.data.id);
  if (callback) {
    if (event.data.status === "success") {
      callback.resolve(event.data.data);
    } else {
      callback.reject(event.data.message);
    }
    callbacks.delete(event.data.id);
  }
}

function getNextWorker() {
  const worker = workers[messageId % numCores];
  return worker;
}

export async function sendToWorker_single(action, data, token, useLocal = true) {
  const text = data.text;
    let chunks;
  if(action === 'create_single'){
   chunks = textSplitter.split(text);
  }
  else chunks = [text];

  const promises = [];

  
  for (let i = 0; i < chunks.length; i++) {
    const promise = new Promise((resolve, reject) => {
      const id = messageId++;
      callbacks.set(id, { resolve, reject });
      data.text = chunks[i];
      const worker = getNextWorker();
      const message = {
        id,
        command: action,
        useLocal,
        token,
        ...data,
      };

      worker.postMessage(message);
    });
    promises.push(promise);
  }

  return Promise.all(promises); // This will resolve when all chunks have been processed
}





export async function sendToWorker(action, data, token, useLocal=true) {
    return new Promise((resolve, reject) => {
        const id = messageId++; // Increment message ID for each new message
        callbacks.set(id, { resolve, reject }); // Store the resolve and reject functions with the unique ID

        // Prepare the message with the unique ID
        const message = {
            id,
            command: action,
            useLocal,
            token,
            ...data,
            
        };


        switch (action) {
            case "initiate":
            case "create":
            case "search":
            case "delete":
            case "update":
                worker_base.postMessage(message);
                break;
            default:
                console.error('Unknown action:', action);
                reject('Unknown action'); // Immediately reject unknown actions
                callbacks.delete(id); // Clean up after handling
        }
    });
}

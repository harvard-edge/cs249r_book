import { generateImageElements, generateVideoElements } from "../../components/generative_page/genAI/searchDuck.js";

export function fetchMaterials(query, subject, requestType) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./worker_duck.js', import.meta.url)); // Create a new worker

        worker.onmessage = async function(e) {
            const { status, type, data } = e.data;

            console.log("status, type, requestType:", status, type, requestType);
            if (status === 'success') {
                try {
                    if (type === 'image') {
                        const imageDivs = await generateImageElements(data); // Await here
                        console.log("imageDivs resolving", imageDivs);
                        resolve({ divs: imageDivs });
                    } else if (type === 'video') {
                        const videoDivs = await generateVideoElements(data); // Await here
                        resolve({ divs: videoDivs });
                    }
                } catch (error) {
                    reject(error);
                } finally {
                    worker.terminate(); // Terminate the worker once the task is done
                }
            } else {
                reject(new Error(e.data.message));
                worker.terminate(); // Terminate the worker in case of error
            }
        };

        worker.onerror = function(error) {
            reject(new Error(error.message));
            worker.terminate(); // Terminate the worker in case of error
        };

        if (requestType === 'image') {
            worker.postMessage({ type: 'image', query: `diagrams about ${query} and ${subject}` });
        } else if (requestType === 'video') {
            worker.postMessage({ type: 'video', query: `${query} and ${subject}` });
        } else {
            reject(new Error('Invalid request type. Please specify "image" or "video".'));
            worker.terminate(); // Terminate the worker in case of invalid request type
        }
    });
}

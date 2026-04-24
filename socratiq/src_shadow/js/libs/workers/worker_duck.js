let getOnlineMaterials //, generateVideoElements;

self.onmessage = async (event) => {
    // Import the functions once if they haven't been imported yet
    if (!getOnlineMaterials) {
        const module = await import ("../../components/generative_page/genAI/searchDuck.js")
        // const module = await import('../agents/duck_search.js'); // Adjust the path as needed
        // generateImageElements = module.generateImageElements;
        // generateVideoElements = module.generateVideoElements;
        getOnlineMaterials = module.getOnlineMaterials
    }

    try {
        let result;
        const { type, query } = event.data;

        switch (type) {
            case 'image':
                result = await getOnlineMaterials(query, "images");
                break;
            case 'video':
                result = await getOnlineMaterials(query, "videos");
                break;
            default:
                throw new Error('Unknown type');
        }

        // Post the result back to the main thread
        self.postMessage({ status: 'success', type: type, data: result });
    } catch (error) {
        self.postMessage({ status: 'error', message: error.message });
    }
};

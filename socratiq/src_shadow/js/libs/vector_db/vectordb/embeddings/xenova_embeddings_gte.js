import { pipeline, env } from '@xenova/transformers';

// Skip initial check for local models, since we are not loading any local models.
env.allowLocalModels = false;

// Due to a bug in onnxruntime-web, we must disable multithreading for now.
// See https://github.com/microsoft/onnxruntime/issues/14445 for more information.
env.backends.onnx.wasm.numThreads = 1;
class PipelineSingleton {
    static instance = null;
    static instancePromise = null;  // Store the promise of initialization

    static async getInstance(progress_callback = null) {
        if (!this.instance) {
            if (!this.instancePromise) {
                // Start the initialization and store the promise
                this.instancePromise = pipeline(
                    "feature-extraction", "Supabase/gte-small",
                    { progress_callback })
                  .then(instance => {
                    this.instance = instance;  // When ready, set the instance
                    return instance;
                  })
                  .catch(error => {
                    console.error("Failed to initialize the model", error);
                    this.instancePromise = null;  // Reset promise to allow retry
                    throw error;
                  });
            }
            // Wait for the existing promise to resolve
            return this.instancePromise;
        }
        return this.instance;
    }
}


export const embed = async (text) => {
    let model;
    try {
        model = await PipelineSingleton.getInstance((data) => {
            // Handle pipeline creation progress
            // console.log('Progress:', data); // this will show progress
        });
    } catch (error) {
        console.error("Error initializing model:", error);
        throw error; // Re-throw to signal error condition to caller
    }

    try {
        const embeddings = await model(text, {
            pooling: "mean",
            normalize: true,
            quantize: true, 
            precision: 'binary' 
        });

        // console.log("inside embeddings", embeddings)
        return await embeddings.data;
    } catch (error) {
        console.error("Error generating embeddings:", error);
        // Optionally, handle the error by returning a default value or similar
        throw error; // Re-throw to signal error condition to caller
    }
};


// Define the CustomTextSplitter class
export class CustomTextSplitter {
    constructor(interval, overlap, charToSplit = '\n\n') {
        this.interval = interval;
        this.overlap = overlap;
        this.charToSplit = charToSplit;
    }

    split(text) {
        const result = [];
        const lines = text.split(this.charToSplit);

        let currentChunk = [];
        let currentSize = 0;

        lines.forEach(line => {
            const tokens = line.match(/\w+|[^\w\s]+/g) || [];
            tokens.forEach(token => {
                // Check if adding this token would exceed the interval size
                if (currentSize + token.length + 1 > this.interval) {
                    // If so, push the current chunk to the result and prepare a new chunk with overlap
                    result.push(currentChunk.join(' '));
                    // Start new chunk with the last 'overlap' tokens from the current chunk
                    currentChunk = currentChunk.slice(-this.overlap);
                    currentSize = currentChunk.join(' ').length + 1; // Recalculate the size of the new chunk
                }

                // Add the token to the current chunk and update the size
                currentChunk.push(token);
                currentSize += token.length + 1; // Add one for the space that follows the token
            });
        });

        // Add the last chunk if it contains any tokens
        if (currentChunk.length > 0) {
            result.push(currentChunk.join(' '));
        }

        return result;
    }
}

// // Example text to split
// const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur tincidunt magna ut justo gravida, sed gravida nisi euismod. Vivamus nec dictum libero.";

// // Instantiate the text splitter with specific interval and overlap values
// const intervalSize = 50; // Example interval size, can be adjusted to your needs
// const overlapSize = 5; // Example overlap size, adjust as needed

// const textSplitter = new CustomTextSplitter(intervalSize, overlapSize);

// // Split the text and print each chunk
// const chunks = textSplitter.split(text);
// console.log("Chunks of text:");
// chunks.forEach((chunk, index) => {
//     console.log(`Chunk ${index + 1}: ${chunk}`);
// });

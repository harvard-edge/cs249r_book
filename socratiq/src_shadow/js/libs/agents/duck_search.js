export async function textImageVideoSearch(query, type) {
    const baseUrl = "https://duckducksearch.azurewebsites.net/api/http_function";
    const validTypes = ['text', 'images', 'videos'];

    if (!validTypes.includes(type)) {
        console.error(`Invalid type provided. Please use one of the following: ${validTypes.join(', ')}`);
        return;
    }

    const url = `${baseUrl}/${type}`;
    const headers = {
        'Content-Type': 'application/json'
    };
    const body = JSON.stringify({ query });

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: body
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json(); // Parse the JSON directly

        // console.log(`Results from ${type} search:`);
        console.log(data);
        return data
        
    } catch (error) {
        console.error(`An error occurred while contacting ${url}: ${error}`);
        return error
    }
}

// Example usage
// const searchQuery = "how generative ai works";
// const searchType = "videos"; // Can be "text", "images", or "videos"
// textImageVideoSearch(searchQuery, searchType);

let lastGeneratedContent = {
    base64Content: '',
    timestamp: '',
    hash: ''
};

export function saveGeneratedHashContent(base64Content, timestamp, hash) {
    lastGeneratedContent = {
        base64Content,
        timestamp,
        hash,
        contentLength: base64Content.length,
        contentStart: base64Content.substring(0, 50),
        contentEnd: base64Content.substring(base64Content.length - 50)
    };
    console.log('Saved Generated Content:', lastGeneratedContent);
}

export function compareWithLastGenerated(verifyContent, verifyTimestamp) {
    try {
        const parsedGenerated = JSON.parse(lastGeneratedContent.base64Content);
        const parsedVerified = JSON.parse(verifyContent);

        console.log('Comparing Hash Contents:', {
            contentLengthsMatch: lastGeneratedContent.base64Content.length === verifyContent.length,
            timestampMatch: lastGeneratedContent.timestamp === verifyTimestamp,
            contentMatch: lastGeneratedContent.base64Content === verifyContent,
            generatedContent: parsedGenerated,
            verifiedContent: parsedVerified,
            // Show differences if content doesn't match
            differences: lastGeneratedContent.base64Content !== verifyContent ? 
                findFirstDifference(lastGeneratedContent.base64Content, verifyContent) : null
        });
    } catch (error) {
        console.error('Error comparing contents:', error);
        console.log('Generated content:', lastGeneratedContent.base64Content);
        console.log('Verified content:', verifyContent);
    }
}

function findFirstDifference(str1, str2) {
    const minLength = Math.min(str1.length, str2.length);
    for (let i = 0; i < minLength; i++) {
        if (str1[i] !== str2[i]) {
            return {
                position: i,
                generated: str1.substring(i - 10, i + 10),
                verified: str2.substring(i - 10, i + 10)
            };
        }
    }
    return null;
}
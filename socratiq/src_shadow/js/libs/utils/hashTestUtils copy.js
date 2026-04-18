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

export function compareWithLastGenerated(verifyBase64Content, verifyTimestamp) {
    console.log('Comparing Hash Contents:', {
        contentLengthsMatch: lastGeneratedContent.base64Content.length === verifyBase64Content.length,
        timestampMatch: lastGeneratedContent.timestamp === verifyTimestamp,
        contentStartMatch: lastGeneratedContent.contentStart === verifyBase64Content.substring(0, 50),
        contentEndMatch: lastGeneratedContent.contentEnd === verifyBase64Content.substring(verifyBase64Content.length - 50),
        // Log exact differences if content lengths match but content differs
        firstDifferentChar: findFirstDifference(lastGeneratedContent.base64Content, verifyBase64Content)
    });
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
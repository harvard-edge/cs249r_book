import {SIZE_LIMIT_LLM_CALL} from '../../../configs/env_configs'
export function reduceTextSize(text, sizeLimit = SIZE_LIMIT_LLM_CALL) {
    if (text.length <= sizeLimit) {
        return text;
    }

    const words = text.split(' ');
    let result = '';
    const chunkSize = Math.ceil(words.length / 10); // Remove ~10% at a time

    while (result.length < sizeLimit && words.length > 0) {
        const startIndex = Math.floor(Math.random() * words.length);
        const endIndex = Math.min(startIndex + chunkSize, words.length);
        const chunk = words.splice(startIndex, endIndex - startIndex).join(' ');

        if (result.length + chunk.length <= sizeLimit) {
            result += (result ? ' ' : '') + chunk;
        }
    }

    return result.trim();
}
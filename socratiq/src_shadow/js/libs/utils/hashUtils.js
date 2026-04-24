import CryptoJS from 'crypto-js';
import { APP_SECRET } from '../../../configs/env_configs.js';

export function uint8ArrayToBase64(uint8Array) {
    let binary = '';
    const chunk = 8192;
    for (let i = 0; i < uint8Array.length; i += chunk) {
        binary += String.fromCharCode.apply(
            null,
            uint8Array.subarray(i, i + chunk)
        );
    }
    return btoa(binary);
}

export function generateContentHash(base64Content, timestamp) {
    const fixedSecret = APP_SECRET;

    const hashInput = `${base64Content}${fixedSecret}${timestamp}`;
    


    return CryptoJS.SHA256(hashInput)
        .toString(CryptoJS.enc.Hex)
        .substring(0, 16);
}
import { resizeImage } from './utils.js';

let mediaStream = null;

export async function startCamera() {
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraContainer = document.getElementById('cameraContainer');
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        cameraVideo.srcObject = mediaStream;
        cameraContainer.style.display = 'block';
    } catch (err) {
        console.error("Camera access denied:", err);
        alert("Could not access camera. Please check permissions.");
    }
}

export function stopCamera() {
    const cameraContainer = document.getElementById('cameraContainer');
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    cameraContainer.style.display = 'none';
}

export function setupCameraEvents() {
    // Avatar camera events removed
}

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
    const btnCamera = document.getElementById('btnCamera');
    const btnStopCamera = document.getElementById('btnStopCamera');
    const btnSnap = document.getElementById('btnSnap');
    const btnUpload = document.getElementById('btnUpload');
    const fileInput = document.getElementById('fileInput');
    const profileAvatarUrlInput = document.getElementById('profileAvatarUrl');
    const avatarPreview = document.getElementById('avatarPreview');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraCanvas = document.getElementById('cameraCanvas');

    if (profileAvatarUrlInput) {
        profileAvatarUrlInput.addEventListener('input', () => {
            avatarPreview.src = profileAvatarUrlInput.value || ''; 
        });
    }

    if (btnUpload && fileInput) {
        btnUpload.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    const result = await resizeImage(file, 300, 300, 0.7);
                    profileAvatarUrlInput.value = result;
                    avatarPreview.src = result;
                } catch (err) {
                    console.error("Error resizing image:", err);
                    alert("Failed to process image. Please try another file.");
                }
            }
        });
    }

    if (btnCamera) {
        btnCamera.addEventListener('click', startCamera);
    }
    
    if (btnStopCamera) {
        btnStopCamera.addEventListener('click', stopCamera);
    }

    if (btnSnap) {
        btnSnap.addEventListener('click', () => {
            if (!mediaStream) return;
            
            let w = cameraVideo.videoWidth;
            let h = cameraVideo.videoHeight;
            
            const MAX_DIM = 300;
            if (w > h) {
                if (w > MAX_DIM) {
                    h = Math.round((h * MAX_DIM) / w);
                    w = MAX_DIM;
                }
            } else {
                if (h > MAX_DIM) {
                    w = Math.round((w * MAX_DIM) / h);
                    h = MAX_DIM;
                }
            }

            cameraCanvas.width = w;
            cameraCanvas.height = h;
            
            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraVideo, 0, 0, w, h);
            
            const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.7);
            
            profileAvatarUrlInput.value = dataUrl;
            avatarPreview.src = dataUrl;
            
            stopCamera();
        });
    }
}

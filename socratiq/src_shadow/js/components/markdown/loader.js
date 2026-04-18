const loaderStyles = `
    .skeleton-loader {
        width: 100%;
        max-width: 600px;
        padding: 20px;
    }

    .skeleton-loader > div {
        background-color: #4b5563;
        height: 8px;
        border-radius: 4px;
        margin-bottom: 16px;
        width: 100%;
        animation: pulse 1.5s ease-in-out infinite;
    }

    .skeleton-loader > div:nth-last-child(1) { width: 90%; }
    .skeleton-loader > div:nth-last-child(2) { width: 95%; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .5; }
    }
`;

const loaderHTML = `
    <div role="status" class="skeleton-loader">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
        <div></div>
    </div>
`;

export const htmlLoader = `
    <style>${loaderStyles}</style>
    ${loaderHTML}
`;

export function removeLoader() {
    const loaderElement = document.querySelector('#loader');
    if (loaderElement) {
        loaderElement.remove();
    }
}
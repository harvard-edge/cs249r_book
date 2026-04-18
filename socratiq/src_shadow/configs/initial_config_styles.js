

export function initial_config_styles() {

    document.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(el => {
        el.removeAttribute('class');
        el.removeAttribute('style');
        // You can also set specific styles if needed
        // el.style.fontSize = '32px';
        // el.style.fontWeight = 'bold';
    });
}
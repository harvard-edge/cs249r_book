   // Function to replace elements with a specific class name
   export function replaceElementsWithErrorNotice(className, shadowEle) {
    const elements = shadowEle.querySelectorAll(`.${className}`);

    let found = false;
    elements.forEach(element => {
        const errorNotice = document.createElement('div');
        errorNotice.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative';
        errorNotice.role = 'alert';
        errorNotice.innerHTML = `
            <strong class="font-bold">Error: </strong>
            <span class="block sm:inline">Check your network connection and try again.</span>
        `;
        
        element.replaceWith(errorNotice);
        found = true
    });

    if (found){
        removeElementByHtmlId(className, shadowEle);
    }
}


export function removeElementByHtmlId(id, shadowEle) {
    const elementsToRemove = shadowEle.querySelectorAll(`#${id}`);
    elementsToRemove.forEach(element => {
        element.remove();
    });
}
// import { html as loaderHtml } from './loader.html'
import loaderHtml from './loader.html'  // This expects a default export

export function removeLoader() {
    // Assuming you've given your loader HTML an id or unique class for easy selection
    var loaderElement = document.querySelector('#loader');
    if (loaderElement) {
      loaderElement.remove(); // This removes the loader element from the document
      // Or you might want to replace it with your API content
      // loaderElement.innerHTML = 'Your API content here';
    }
  } 

export const htmlLoader = loaderHtml 

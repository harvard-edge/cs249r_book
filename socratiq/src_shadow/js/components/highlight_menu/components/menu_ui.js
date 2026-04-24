

export function initiate_menu(shadowEle) {

// const textarea = shadowEle.querySelector('#responseTextarea');
// const tabPlaceholder = shadowEle.querySelector('#tabPlaceholder');

// textarea.addEventListener('input', () => {
//   if (textarea.value.length > 0) {
//     tabPlaceholder.classList.remove('hidden');
//   } else {
//     tabPlaceholder.classList.add('hidden');
//   }
// });

shadowEle.querySelector('#citation').addEventListener('click', function() {
  const menu = shadowEle.querySelector('#citationMenu');
  const citationUpBtn =this.querySelector('#up-chevron')
  const citationDwnBtn =this.querySelector('#down-chevron')
  menu.classList.toggle('hidden');
  citationDwnBtn.classList.toggle('hidden')
  citationUpBtn.classList.toggle('hidden')
  // toggleCitations(menu)
});

}

function toggleCitations_off(menu) {
  const citationUpBtn = menu.querySelector('#up-chevron');
  const citationDwnBtn = menu.querySelector('#down-chevron');
  const menu_citations = menu.querySelector('#citationMenu');

  // Add 'hidden' class to citation_menu and citationUpBtn to hide them
  menu_citations.classList.add('hidden');
  citationUpBtn.classList.add('hidden');

  // Remove 'hidden' class from citationDwnBtn to show it
  citationDwnBtn.classList.remove('hidden');
}

export function close_menu(shadowEle) {
  
  const menu = shadowEle.querySelector('#text-selection-menu-highlight');
  toggleCitations_off(menu)


  menu.classList.add('hidden');
}
/* Shared, progressively enhanced views for one existing numbered figure. */
(()=>{
  'use strict';
  let serial=0;
  function enhance({figure,diagram,explore,initialView='diagram'}){
    if(!figure||!diagram||!explore||!figure.contains(diagram)||!figure.contains(explore))throw new Error('Figure views require two panels in one figure.');
    if(figure.dataset.figureViewsReady==='true')return;
    const prefix=(figure.id||figure.closest('[id]')?.id||'mlsys-figure')+'-views-'+(++serial);
    const views={diagram,explore};
    const tabs=document.createElement('div');
    tabs.className='mlsys-figure-tabs';tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Figure views');
    const buttons={};
    for(const [name,panel] of Object.entries(views)){
      panel.id=panel.id||prefix+'-'+name;
      panel.classList.add('mlsys-'+name+'-panel');
      panel.setAttribute('role','tabpanel');
      const button=document.createElement('button');
      button.type='button';button.id=prefix+'-'+name+'-tab';button.setAttribute('role','tab');
      button.setAttribute('aria-controls',panel.id);button.textContent=name==='diagram'?'Diagram':'Explore';
      panel.setAttribute('aria-labelledby',button.id);button.addEventListener('click',()=>select(name));
      button.addEventListener('keydown',event=>{
        const key=event.key;if(!['ArrowLeft','ArrowRight','Home','End'].includes(key))return;
        event.preventDefault();const target=key==='Home'?'diagram':key==='End'?'explore':name==='diagram'?'explore':'diagram';
        select(target);buttons[target].focus();
      });
      buttons[name]=button;tabs.append(button);
    }
    function select(name){
      for(const [view,panel] of Object.entries(views)){
        const active=view===name;panel.hidden=!active;buttons[view].setAttribute('aria-selected',String(active));buttons[view].tabIndex=active?0:-1;
      }
      figure.dataset.figureView=name;
      figure.dispatchEvent(new CustomEvent('mlsys-view-change',{detail:{view:name},bubbles:false}));
    }
    figure.classList.add('mlsys-figure-views');figure.dataset.figureViewsReady='true';figure.insertBefore(tabs,figure.firstChild);
    select(initialView==='explore'?'explore':'diagram');
    return {select};
  }
  // Quarto clones local figures for cross-reference hover previews. Keep those
  // clones static: controls have no listeners there and copied IDs are invalid.
  const sanitizePreviews = () => {
    document.querySelectorAll('.tippy-content .mlsys-figure-views').forEach(preview => {
      preview.querySelectorAll('.mlsys-figure-tabs,.mlsys-explore-panel').forEach(node => node.remove());
      preview.querySelectorAll('.mlsys-diagram-panel').forEach(panel => {
        panel.hidden = false; panel.removeAttribute('role'); panel.removeAttribute('aria-labelledby');
      });
      preview.classList.remove('mlsys-figure-views');
      for (const node of [preview, ...preview.querySelectorAll('[id]')]) node.removeAttribute('id');
      preview.querySelectorAll('[aria-controls],[aria-labelledby],[aria-describedby]').forEach(node => {
        node.removeAttribute('aria-controls'); node.removeAttribute('aria-labelledby'); node.removeAttribute('aria-describedby');
      });
    });
  };
  const watchPreviews = () => {
    const observer = new MutationObserver(records => {
      if (records.some(record => record.addedNodes.length)) sanitizePreviews();
    });
    observer.observe(document.body, {childList:true, subtree:true});
    sanitizePreviews();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', watchPreviews, {once:true});
  else watchPreviews();
  window.MLSysFigureViews=Object.freeze({enhance});
})();

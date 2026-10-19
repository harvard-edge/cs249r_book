/* fig-tiling-diagram: preserve B above C, A left of C, and green/pink/blue roles.
 * Supplementary toy case: A 4×4, B 4×6, C 4×6; one 2×2 output tile,
 * two K-tiles. Explicit copies, retained partial sums, and one final write.
 */
(() => {
  'use strict';
  window.MLSysFigureLessons = window.MLSysFigureLessons || {};
  window.MLSysFigureLessons['matrix-tiling'] = root => {
    const abort = new AbortController();
    const on = (target, type, fn) => target.addEventListener(type, fn, {signal:abort.signal});
    const make = (tag, cls, text) => { const el = document.createElement(tag); if(cls) el.className=cls; if(text !== undefined) el.textContent=text; return el; };
    const stages=['Choose output tile','Load first operand tiles','Accumulate first product','Load next operand tiles','Finish reduction','Write completed output'];
    const A=[[9,10],[13,14]], A2=[[11,12],[15,16]], B=[[1,2],[3,4]], B2=[[5,6],[7,8]];
    const multiply=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,value,k)=>sum+value*b[k][j],0)));
    const first=multiply(A,B), second=multiply(A2,B2), final=first.map((row,i)=>row.map((v,j)=>v+second[i][j]));
    let stage=0, running=false, timer=null, destroyed=false;
    const intro=make('p','mlsys-tiling-intro','Follow one blue output tile through two inner-dimension blocks. This small worked case uses A: 4 × 4, B: 4 × 6, and 2 × 2 tiles; indices start at zero, and other matrix values are omitted.');
    const controls=make('div','mlsys-tiling-controls'), buttons={};
    for(const name of ['Play','Back','Next','Restart']) { const button=make('button','',name); button.type='button'; buttons[name.toLowerCase()]=button; controls.append(button); }
    const label=make('label','','Step '), menu=make('select'); menu.id=root.id+'-sequence'; label.htmlFor=menu.id;
    stages.forEach((name,i)=>{const option=make('option','',`${i+1}. ${name}`); option.value=i; menu.append(option);}); label.append(menu); controls.append(label);
    const diagram=make('div','mlsys-tiling-diagram');
    const matrices={};
    for(const [name,rows,cols,role] of [['A',4,4,'left'],['B',4,6,'right'],['C',4,6,'output']]) {
      const panel=make('section','mlsys-tiling-matrix mlsys-tiling-'+name.toLowerCase());
      const heading=make('h4','',`${name} matrix · ${rows} × ${cols}`); const grid=make('div','mlsys-tiling-grid'); grid.style.gridTemplateColumns=`repeat(${cols},1fr)`;
      const cells=[]; for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const cell=make('span'); grid.append(cell); cells.push({cell,r,c});}
      const detail=make('p'); panel.append(heading,grid,detail); diagram.append(panel); matrices[name]={panel,cells,detail,role};
    }
    const memoryLabel=make('p','mlsys-tiling-memory-label','Fast on-chip storage · operand copies and accumulators');
    const memory=make('div','mlsys-tiling-memory'), cards={};
    for(const [key,title] of [['a','A operand copy'],['b','B operand copy'],['c','C accumulator']]) {
      const card=make('section','mlsys-tiling-card mlsys-tiling-'+key+'-card'), heading=make('h4','',title), state=make('p','mlsys-tiling-location'), table=make('table'), body=make('tbody');
      table.setAttribute('aria-label',title+' values'); table.append(body); const cells=[]; for(let r=0;r<2;r++){const row=make('tr');for(let c=0;c<2;c++){const cell=make('td');cells.push(cell);row.append(cell);}body.append(row);}card.append(heading,state,table);memory.append(card);cards[key]={state,cells};
    }
    const status=make('div','mlsys-tiling-status'); status.setAttribute('aria-live','polite');status.setAttribute('aria-atomic','true');
    const narration=make('p'), arithmetic=make('p','mlsys-tiling-arithmetic'), traffic=make('p','mlsys-tiling-traffic');status.append(narration,arithmetic,traffic);
    const caveat=make('p','mlsys-tiling-footnote','Counts describe this tile schedule only: scalar elements, not bytes or measured DRAM transactions. Each loaded operand is reused in two multiply-accumulates. Capacity, cache behavior, tile shape, and hardware determine actual performance; tiling does not guarantee peak throughput.');
    root.append(intro,controls,make('p','mlsys-tiling-memory-label','Backing memory · source operands and output'),diagram,memoryLabel,memory,status,caveat);
    const fill=(key,values)=>cards[key].cells.forEach((cell,index)=>{cell.textContent=values?values[Math.floor(index/2)][index%2]:'—';});
    function render(){
      const k=stage>=3?1:0, loaded=stage>=1, sums=stage>=4?final:stage>=2?first:[[0,0],[0,0]];
      for(const [name,matrix] of Object.entries(matrices)){
        matrix.cells.forEach(({cell,r,c})=>{
          const band=name==='A'?r>=2:name==='B'?c>=2&&c<4:r>=2&&c>=2&&c<4;
          const active=name==='A'?r>=2&&Math.floor(c/2)===k:name==='B'?c>=2&&c<4&&Math.floor(r/2)===k:band;
          cell.classList.toggle('mlsys-tiling-band',band); cell.classList.toggle('mlsys-tiling-active',active);
          cell.classList.toggle('mlsys-tiling-written',name==='C'&&active&&stage===5);
          cell.textContent=active?(name==='A'?(k?A2:A)[r-2][c%2]:name==='B'?(k?B2:B)[r%2][c-2]:stage===5?final[r-2][c-2]:''):'';
        });
      }
      matrices.A.detail.textContent=`Mtile × Ktile: rows 2–3, columns ${k*2}–${k*2+1}`;
      matrices.B.detail.textContent=`Ktile × Ntile: rows ${k*2}–${k*2+1}, columns 2–3`;
      matrices.C.detail.textContent=stage===5?'Completed tile written to backing memory':'Mtile × Ntile: rows 2–3, columns 2–3';
      fill('a',loaded&&stage<5?(k?A2:A):null);fill('b',loaded&&stage<5?(k?B2:B):null);fill('c',stage===5?null:sums);
      cards.a.state.textContent=cards.b.state.textContent=stage===5?'Released after output write':loaded?`Copied from backing memory · k block ${k}`:'Not loaded';
      cards.c.state.textContent=stage===5?'Written back; local storage released':stage>=2?'Retained on chip across K blocks':'Initialized to zero on chip';
      narration.textContent=[
        'Choose the blue C tile and initialize its four accumulators to zero on chip. The green A row band and pink B column band identify all operands needed across K.',
        'Copy the first green A block and pink B block into fast memory. Their source matrices remain in backing memory. The blue accumulators stay on chip.',
        'Multiply the resident tiles and add their product to C. Each of the four A values participates in two output columns; each B value participates in two output rows. These are partial sums, not final outputs.',
        'Replace the operand copies with the next K blocks. Retain the blue partial sums unchanged on chip: no intermediate C write or reload is needed in this schedule.',
        'Add the second tile product to the retained partial sums. Both K blocks have now contributed, so the output tile is complete.',
        'Write the four completed C values to backing memory once. Local operand copies and accumulators can now be reused for another output tile. The arithmetic is unchanged; staging concentrates reuse in fast memory.'
      ][stage];
      arithmetic.textContent=stage<2?'For each K block: Ctile ← Ctile + Atile × Btile.':stage===2||stage===3?'Top-left output: 0 + 9 × 1 + 10 × 3 = 39 (partial).': 'Top-left output: 39 + 11 × 5 + 12 × 7 = 178 (complete).';
      if(stage===3) arithmetic.textContent='Retained first-block sum: 0 + 9 × 1 + 10 × 3 = 39. The new operand tiles have not contributed yet.';
      if(stage===5) arithmetic.textContent+=' Final tile: [[178, 220], [242, 300]].';
      traffic.textContent=`This output tile: ${stage>=3?16:stage>=1?8:0} operand elements copied in · ${stage===5?4:0} output elements written back · ${stage>=4?16:stage>=2?8:0} multiply-accumulates completed.`;
      menu.value=stage;buttons.back.disabled=stage===0;buttons.next.disabled=stage===5;buttons.play.textContent=running?'Pause':stage===5?'Replay':'Play';
    }
    function pause(){running=false;clearTimeout(timer);timer=null;if(!destroyed)render();}
    function schedule(){timer=setTimeout(()=>{if(!running||destroyed)return;stage++;if(stage===5)running=false;render();if(running)schedule();},7500);}
    on(buttons.play,'click',()=>{if(running){pause();return;}if(stage===5)stage=0;running=true;render();schedule();});
    on(buttons.back,'click',()=>{pause();stage=Math.max(0,stage-1);render();});on(buttons.next,'click',()=>{pause();stage=Math.min(5,stage+1);render();});on(buttons.restart,'click',()=>{pause();stage=0;render();});
    on(menu,'change',()=>{const next=Number(menu.value);pause();stage=next;render();});on(document,'visibilitychange',()=>{if(document.hidden)pause();});
    const observer=typeof IntersectionObserver==='undefined'?null:new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&running)pause();},{threshold:.15});observer?.observe(root);
    render();return{pause,destroy(){if(destroyed)return;pause();destroyed=true;observer?.disconnect();abort.abort();root.replaceChildren();}};
  };
})();

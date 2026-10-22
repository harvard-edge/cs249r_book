(function () {
  const embedded = document.getElementById('adp-data');
  const svg = document.querySelector('.adp-map');
  if (!embedded) return;
  const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const list = document.getElementById('crs-list'), empty = document.getElementById('crs-empty');
  const q = document.getElementById('crs-q'), chips = [...document.querySelectorAll('.crs-chip')];
  let rows = [], readers = [], early = [], schools = [], kind = 'all', view = 'courses';
  const tabs = [...document.querySelectorAll('.crs-tab')];
  const intro = document.getElementById('crs-intro');
  // "IIT" should find "Indian Institute of Technology, Madras": match on the
  // name's initials as well as the name.
  const initials = n => n.replace(/[(),]/g, ' ').split(/\s+/)
    .filter(w => w && !/^(of|the|and|at|de|da|di|la|e|in|for)$/i.test(w)).map(w => w[0]).join('').toLowerCase();
  const hayOf = (...parts) => { const t = parts.filter(Boolean).join(' '); return (t + ' ' + initials(t)).toLowerCase(); };
  const rBox = document.getElementById('crs-readers');
  const ADD = 'https://github.com/harvard-edge/cs249r_book/issues/new?template=add_course.yml';

  const mark = s => `<span class="adp-mark${s.logo ? '' : ' adp-mark--mono'}">${
    s.logo ? `<img src="${esc(s.logo)}" alt="" loading="lazy" width="28" height="28">` : esc(s.mono)}</span>`;

  function project(lat, lon) {
    const k = +svg.dataset.k, ox = +svg.dataset.ox, oy = +svg.dataset.oy;
    const A1 = 1.340264, A2 = -0.081106, A3 = 0.000893, A4 = 0.003796, M = Math.sqrt(3) / 2;
    const t = Math.asin(M * Math.sin(lat * Math.PI / 180)), t2 = t * t, t6 = t2 * t2 * t2;
    const x = (lon * Math.PI / 180) * Math.cos(t) / (M * (A1 + 3 * A2 * t2 + t6 * (7 * A3 + 9 * A4 * t2)));
    return [ox + x * k, oy - t * (A1 + A2 * t2 + t6 * (A3 + A4 * t2)) * k];
  }

  // Schools view: every school we know of, A to Z within each country, so a
  // student can see at a glance whether theirs is here. Teaching schools link
  // back to their courses; reader schools are marked as such.
  function drawSchools(needle) {
    document.getElementById('directory-title').parentElement.hidden = false;
    document.querySelector('.crs-mapwrap').hidden = false;
    const shown = schools.filter(x => !needle || x.hay.includes(needle));
    const groups = new Map();
    shown.forEach(x => { if (!groups.has(x.country)) groups.set(x.country, []); groups.get(x.country).push(x); });
    const nc = new Set(schools.map(x => x.country)).size;
    intro.hidden = false;
    // Counts follow the complete refreshed list, independent of search filters.
    intro.innerHTML = `${schools.length} institutions across ${nc} countries, represented through instructor-submitted courses, public syllabi, GitHub affiliations, and referrals from university websites. ` +
      `Don't see yours? <a href="${ADD}" target="_blank" rel="noopener">Add a course</a> or visit the <a href="https://mlsysbook.ai/instructors/">Instructor Hub</a>.`;
    list.innerHTML = [...groups.entries()]
      .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
      .map(([country, xs]) => `<h2 class="crs-country">${esc(country)} <span>${xs.length}</span></h2><ul class="crs-schools">` +
        xs.sort((a, b) => a.name.localeCompare(b.name)).map(x => {
          const inner = `${mark(x)}<span class="crs-sname">${esc(x.name)}</span>` +
            `<span class="crs-badge crs-badge--${x.status}">${x.status === 'teach' ? 'Course listed' : x.status === 'early' ? 'Early adopter' : 'Community engagement'}</span>`;
          return x.status === 'teach'
            ? `<li><button type="button" class="crs-srow" data-name="${esc(x.name)}">${inner}</button></li>`
            : `<li><div class="crs-srow">${inner}</div></li>`;
        }).join('') + `</ul>`).join('');
    rBox.hidden = true;
    document.getElementById('crs-early').hidden = true;
    empty.hidden = shown.length > 0;
    empty.innerHTML = `“${esc(q.value.trim())}” isn't on the list yet. If a course there uses the book, <a href="${ADD}" target="_blank" rel="noopener">add it</a>.`;
  }

  function draw() {
    const needle = q.value.trim().toLowerCase();
    const featured = document.getElementById('hosted-courses');
    const cards = [...document.querySelectorAll('.course-feature')];
    cards.forEach(card => {
      card.hidden = view === 'schools' || (kind !== 'all' && card.dataset.kind !== kind) ||
        (needle && !card.dataset.search.includes(needle));
    });
    featured.hidden = cards.every(card => card.hidden);
    document.getElementById('crs-chips').hidden = view === 'schools';
    tabs.forEach(t => t.setAttribute('aria-selected', String(t.dataset.view === view)));
    document.getElementById('directory-heading').textContent = view === 'schools'
      ? 'Where people are learning.' : 'Find your course.';
    document.getElementById('directory-description').textContent = view === 'schools'
      ? 'Explore the institutions connected to the MLSysBook learning community, through teaching and engagement with the book and its resources.'
      : 'Explore ML systems courses, hands-on studios, and teaching archives. Start with a course hosted here, or find an offering at your institution.';
    document.getElementById('directory-title').textContent = view === 'schools'
      ? 'Our learning community' : 'Across the community';
    document.querySelector('#directory-title + span').textContent = view === 'schools'
      ? 'Institutions represented in teaching & engagement' : 'Courses & workshops using MLSysBook';
    if (view === 'schools') return drawSchools(needle);
    intro.hidden = true;
    const shown = rows.filter(r => (kind === 'all' || r.kind === kind) &&
      (!needle || r.hay.includes(needle)));
    document.getElementById('directory-title').parentElement.hidden = shown.length === 0;
    document.querySelector('.crs-mapwrap').hidden = shown.length === 0;
    // Grouped by country, largest first: most visitors scan for their own.
    const groups = new Map();
    shown.forEach(r => { if (!groups.has(r.s.country)) groups.set(r.s.country, []); groups.get(r.s.country).push(r); });
    list.innerHTML = [...groups.entries()]
      .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
      .map(([country, rs]) => `<h2 class="crs-country">${esc(country)} <span>${rs.length}</span></h2>` + rs.map(r => `
      <a class="crs-row" href="${esc(r.url)}" target="_blank" rel="noopener">
        ${mark(r.s)}
        <span class="crs-course"><b>${esc(r.title)}</b>${r.term ? `<span class="crs-term">${esc(r.term)}</span>` : ''}</span>
        <span class="crs-school">${esc(r.s.name)}<span>${esc(r.s.city)}, ${esc(r.s.country)}</span></span>
        <span class="crs-kind crs-kind--${r.kind}">${r.kind === 'workshop' ? 'Workshop' : 'Course'}</span>
        <span class="crs-go" aria-hidden="true">&#8599;</span>
      </a>`).join('')).join('');
    // Readers: only on a search, and only by school name, so typing a
    // country lists its courses without a wall of reader schools.
    const hits = needle.length >= 3 ? readers.filter(r => r.hay.includes(needle)).slice(0, 12) : [];
    rBox.hidden = !hits.length;
    if (hits.length) {
      const one = hits.length === 1 ? hits[0].name : null;
      document.getElementById('crs-readers-title').textContent = shown.length
        ? 'Readers study here too'
        : (one ? `Readers at ${one} use the book` : 'Readers at these schools use the book');
      document.getElementById('crs-readers-note').innerHTML = shown.length
        ? 'These schools have readers but no listed course yet.'
        : `No course there is listed yet, but readers study there. If a class uses the book, <a href="${ADD}" target="_blank" rel="noopener">add it</a>; if not, <a href="https://mlsysbook.ai/instructors/">point your instructor to the Instructor Hub</a>.`;
      document.getElementById('crs-readers-list').innerHTML =
        hits.map(r => `<li>${esc(r.name)}<small>${esc(r.country)}</small></li>`).join('');
    }
    // Early adopters: shown in full when browsing, filtered when searching.
    const eShown = early.filter(e => !needle || e.hay.includes(needle));
    document.getElementById('crs-early').hidden = !eShown.length || !['all', 'course'].includes(kind);
    document.getElementById('crs-early-list').innerHTML = eShown.map(({s}) =>
      `<li>${mark(s)}<span>${esc(s.name)}<small>${esc(s.city)}, ${esc(s.country)}</small></span></li>`).join('');
    empty.hidden = shown.length > 0 || hits.length > 0 || (eShown.length > 0 && ['all', 'course'].includes(kind)) || !featured.hidden;
    empty.innerHTML = `Nothing for “${esc(q.value.trim())}” yet. Know a course that uses the book? <a href="${ADD}" target="_blank" rel="noopener">Add it</a>.`;
  }

  function render(data) {
    const t = data.teaching || [];
    rows = [];
    t.forEach(s => (s.courses || []).forEach(c => rows.push({
      s, kind: s.kind === 'workshop' ? 'workshop' : 'course', title: c.title, term: c.term, url: c.url,
      hay: hayOf(s.name, s.short) + ' ' + [s.city, s.country, c.title, c.term].join(' ').toLowerCase()
    })));
    rows.sort((a, b) => a.s.name.localeCompare(b.s.name));
    readers = (data.readers || []).map(r => ({...r, hay: hayOf(r.name)}));

    early = t.filter(s => s.kind === 'acknowledged').map(s => ({s, hay: hayOf(s.name, s.short) + ' ' + [s.city, s.country].join(' ').toLowerCase()}));
    const seen = new Set();
    schools = t.map(s => ({name: s.name, country: s.country, logo: s.logo, mono: s.mono,
                           status: s.kind === 'acknowledged' ? 'early' : 'teach', hay: hayOf(s.name, s.short) + ' ' + s.country.toLowerCase()}))
      .concat((data.readers || []).map(r => ({name: r.name, country: r.country, logo: r.logo, mono: initials(r.name).slice(0, 4).toUpperCase(),
                                              status: 'readers', hay: hayOf(r.name) + ' ' + r.country.toLowerCase()})))
      .filter(x => { const k = x.name.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });
    draw();

    if (svg) {
      svg.querySelector('.adp-pins').innerHTML = t.slice().sort((a, b) => b.latlon[0] - a.latlon[0]).map(s => {
        const [x, y] = project(s.latlon[0], s.latlon[1]);
        return `<g class="adp-pin adp-pin--${s.kind}" role="button" tabindex="0" aria-label="Filter courses at ${esc(s.name)}" data-name="${esc(s.name)}" transform="translate(${x.toFixed(1)},${y.toFixed(1)})">` +
               `<title>${esc(s.name)}, ${esc(s.country)}</title><circle r="6"/></g>`;
      }).join('');
    }
  }

  function setView(v, push) {
    view = v === 'schools' ? 'schools' : 'courses';
    if (push) {
      try { const u = new URL(location.href); view === 'schools' ? u.searchParams.set('view', 'schools') : u.searchParams.delete('view'); history.replaceState(null, '', u); } catch (e) {}
    }
    draw();
  }
  tabs.forEach(t => t.addEventListener('click', () => setView(t.dataset.view, true)));
  list.addEventListener('click', e => {
    const b = e.target.closest('button.crs-srow');
    if (!b) return;
    q.value = b.dataset.name; setView('courses', true);
    list.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
  try { if (new URLSearchParams(location.search).get('view') === 'schools') view = 'schools'; } catch (e) {}

  // ?school= pre-fills the search, so "find your school" can be shared as a
  // link. (Not ?q=, which Quarto's site search claims for its own highlight.)
  try { const pre = new URLSearchParams(location.search).get('school'); if (pre) q.value = pre; } catch (e) {}
  q.addEventListener('input', draw);
  chips.forEach(c => c.addEventListener('click', () => {
    kind = c.dataset.kind;
    chips.forEach(x => x.setAttribute('aria-pressed', String(x === c)));
    draw();
  }));
  if (svg) svg.addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.closest('.adp-pin')) {
      e.preventDefault();
      e.target.dispatchEvent(new MouseEvent('click', {bubbles: true}));
    }
  });
  if (svg) svg.addEventListener('click', e => {
    const pin = e.target.closest('.adp-pin');
    if (!pin) return;
    e.preventDefault();
    q.value = pin.dataset.name; draw();
    list.scrollIntoView({behavior: 'smooth', block: 'start'});
  });

  try { render(JSON.parse(embedded.textContent)); } catch (e) { /* shell stays */ }
  fetch('../adopters.json?t=' + Date.now())
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(render)
    .catch(err => console.warn('[mlsysbook] adopters refresh failed, showing build-time list:', err));
})();

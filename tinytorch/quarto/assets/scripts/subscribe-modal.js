<script>
/**
 * Subscribe Modal Component for TinyTorch (Quarto version)
 * Elegant popup subscription form
 */

(function() {
  'use strict';

  function createModalHTML() {
    return `
      <div id="subscribe-modal" class="modal-overlay" style="display: none;">
        <div class="modal-container">
          <button class="modal-close" data-close-modal aria-label="Close">&times;</button>
          <div class="modal-content">
            <div class="modal-header">
              <div class="modal-brand-row">
                <span class="modal-brand-item">📚 MLSysBook</span>
              </div>
              <h2 class="modal-title">Stay in the Loop</h2>
              <p class="modal-subtitle">Get updates on new chapters, hands-on labs, and ML systems resources.</p>
            </div>
            <form id="subscribe-modal-form" class="subscribe-form" action="https://buttondown.email/api/emails/embed-subscribe/mlsysbook" method="post">
              <div class="form-row">
                <div class="form-group">
                  <label for="modal-first-name">First name</label>
                  <input type="text" id="modal-first-name" name="metadata__first_name" required placeholder="Jane">
                </div>
                <div class="form-group">
                  <label for="modal-last-name">Last name</label>
                  <input type="text" id="modal-last-name" name="metadata__last_name" required placeholder="Smith">
                </div>
              </div>
              <div class="form-group">
                <label for="modal-email">Email</label>
                <input type="email" id="modal-email" name="email" required placeholder="jane@university.edu">
              </div>
              <div class="form-group">
                <label>I am a...</label>
                <div class="role-options role-options-three-compact">
                  <label class="role-option">
                    <input type="radio" name="metadata__role" value="educator" required>
                    <span class="role-label">👨‍🏫 Educator</span>
                  </label>
                  <label class="role-option">
                    <input type="radio" name="metadata__role" value="student">
                    <span class="role-label">🎓 Student</span>
                  </label>
                  <label class="role-option">
                    <input type="radio" name="metadata__role" value="industry">
                    <span class="role-label">💼 Industry</span>
                  </label>
                </div>
              </div>
              <div class="form-group">
                <label for="modal-organization">Organization <span class="optional-label">(optional)</span></label>
                <input type="text" id="modal-organization" name="metadata__organization" placeholder="University or company">
              </div>
              <div class="form-group">
                <label for="modal-motivation">What brings you here? <span class="optional-label">(optional)</span></label>
                <textarea id="modal-motivation" name="metadata__motivation" rows="2" placeholder="e.g., teaching a course, learning ML systems, building edge devices..."></textarea>
              </div>
              <input type="hidden" name="tag" value="tinytorch-site">
              <button type="submit" class="btn btn-primary subscribe-btn">Subscribe</button>
              <p class="form-note">No spam, ever. Unsubscribe anytime.</p>
            </form>
            <div id="modal-subscribe-success" class="subscribe-success" style="display: none;">
              <div class="success-icon">🎉</div>
              <h3>You're in!</h3>
              <p>Welcome to the ML systems community. We'll keep you updated on new content.</p>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function createModalCSS() {
    const style = document.createElement('style');
    style.textContent = `
      .modal-overlay {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(4px);
        z-index: 10001; align-items: center; justify-content: center;
        padding: 1rem; animation: fadeIn 0.2s ease;
      }
      @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
      @keyframes slideUp { from { opacity: 0; transform: translateY(20px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }
      .modal-container {
        background: white; border-radius: 16px; max-width: 440px; width: 100%;
        max-height: 90vh; overflow-y: auto; position: relative;
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        animation: slideUp 0.3s ease; margin: auto;
      }
      .modal-close {
        position: absolute; top: 1rem; right: 1rem; width: 36px; height: 36px;
        border: none; background: #f8fafc; border-radius: 50%; font-size: 1.5rem;
        color: #64748b; cursor: pointer; display: flex; align-items: center;
        justify-content: center; transition: all 0.2s ease; z-index: 10; line-height: 1;
      }
      .modal-close:hover { background: white; color: #0f172a; transform: scale(1.05); }
      .modal-content { padding: 2rem 2.5rem 2.5rem 2.5rem; }
      .modal-header { text-align: center; margin-bottom: 1.5rem; display: flex; flex-direction: column; align-items: center; }
      .modal-brand-row { display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem; }
      .modal-brand-item { font-size: 0.8rem; font-weight: 600; color: #374151; background: #f3f4f6; padding: 0.3rem 0.6rem; border-radius: 5px; }
      .modal-title { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin: 0 0 0.4rem 0; line-height: 1.2; width: 100%; }
      .modal-subtitle { font-size: 0.9rem; color: #64748b; margin: 0; line-height: 1.5; max-width: 320px; }
      .subscribe-form { display: flex; flex-direction: column; gap: 1.25rem; }
      .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
      .form-row .form-group { min-width: 0; }
      .form-row .form-group input { width: 100%; box-sizing: border-box; }
      .form-group { display: flex; flex-direction: column; gap: 0.5rem; }
      .form-group label { font-size: 0.9rem; font-weight: 600; color: #0f172a; }
      .optional-label { font-weight: 400; color: #64748b; }
      .form-group input[type="text"], .form-group input[type="email"], .form-group textarea {
        padding: 0.875rem 1rem; border: 1px solid #cbd5e1; border-radius: 8px;
        font-size: 1rem; transition: all 0.2s ease; background: #f8fafc; font-family: inherit;
      }
      .form-group textarea { resize: vertical; min-height: 60px; }
      .form-group input:focus, .form-group textarea:focus {
        outline: none; border-color: #3b82f6; background: white;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
      }
      .form-group input::placeholder, .form-group textarea::placeholder { color: #94a3b8; }
      .role-options { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
      .role-options-three-compact { grid-template-columns: repeat(3, 1fr); }
      .role-option { cursor: pointer; }
      .role-option input[type="radio"] { position: absolute; opacity: 0; width: 0; height: 0; }
      .role-label {
        display: flex; align-items: center; justify-content: center; gap: 0.5rem;
        padding: 0.75rem 1rem; border: 2px solid #e2e8f0; border-radius: 8px;
        font-size: 0.9rem; font-weight: 500; color: #475569; transition: all 0.2s ease; background: #f8fafc;
      }
      .role-options-three-compact .role-label { padding: 0.625rem 0.5rem; font-size: 0.8rem; text-align: center; }
      .role-option input[type="radio"]:checked + .role-label { border-color: #3b82f6; background: rgba(59, 130, 246, 0.08); color: #3b82f6; }
      .role-option:hover .role-label { border-color: #cbd5e1; background: white; }
      .subscribe-btn { width: 100%; padding: 1rem; font-size: 1rem; margin-top: 0.5rem;
        display: inline-flex; align-items: center; justify-content: center;
        border-radius: 8px; text-decoration: none; font-weight: 600; border: none; cursor: pointer;
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); color: white; font-family: inherit;
      }
      .subscribe-btn:hover { background: linear-gradient(135deg, #fb923c 0%, #f97316 100%); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3); }
      .form-note { text-align: center; font-size: 0.85rem; color: #64748b; margin: 0; }
      .subscribe-success { text-align: center; padding: 2rem 1rem; }
      .success-icon { font-size: 3rem; margin-bottom: 1rem; }
      .subscribe-success h3 { font-size: 1.5rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem; }
      .subscribe-success p { color: #475569; font-size: 1rem; }

      /* Quarto dark mode */
      .quarto-dark .modal-container, html[data-bs-theme="dark"] .modal-container { background: #1e293b; }
      .quarto-dark .modal-close, html[data-bs-theme="dark"] .modal-close { background: #0f172a; color: #94a3b8; }
      .quarto-dark .modal-title, html[data-bs-theme="dark"] .modal-title,
      .quarto-dark .form-group label, html[data-bs-theme="dark"] .form-group label { color: #f1f5f9; }
      .quarto-dark .modal-subtitle, html[data-bs-theme="dark"] .modal-subtitle { color: #cbd5e1; }
      .quarto-dark .form-group input, html[data-bs-theme="dark"] .form-group input,
      .quarto-dark .form-group textarea, html[data-bs-theme="dark"] .form-group textarea { background: #0f172a; border-color: #334155; color: #f1f5f9; }
      .quarto-dark .role-label, html[data-bs-theme="dark"] .role-label { background: #0f172a; border-color: #334155; color: #cbd5e1; }

      @media (max-width: 640px) { .modal-content { padding: 2rem 1.5rem; } .form-row { grid-template-columns: 1fr; } }
    `;
    return style;
  }

  function initModal() {
    document.head.appendChild(createModalCSS());
    const modalDiv = document.createElement('div');
    modalDiv.innerHTML = createModalHTML();
    document.body.appendChild(modalDiv.firstElementChild);

    const modal = document.getElementById('subscribe-modal');
    const form = document.getElementById('subscribe-modal-form');
    const success = document.getElementById('modal-subscribe-success');

    window.openSubscribeModal = function() {
      modal.style.display = 'flex';
      document.body.style.overflow = 'hidden';
      setTimeout(() => { const fi = document.getElementById('modal-first-name'); if (fi) fi.focus(); }, 100);
    };

    window.closeSubscribeModal = function() {
      modal.style.display = 'none';
      document.body.style.overflow = '';
      setTimeout(() => { form.style.display = 'flex'; form.reset(); success.style.display = 'none'; }, 300);
    };

    modal.addEventListener('click', (e) => { if (e.target === modal) closeSubscribeModal(); });
    const closeBtn = modal.querySelector('[data-close-modal]');
    if (closeBtn) closeBtn.addEventListener('click', closeSubscribeModal);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.style.display === 'flex') closeSubscribeModal(); });

    form.addEventListener('submit', function() {
      setTimeout(() => { form.style.display = 'none'; success.style.display = 'block'; setTimeout(closeSubscribeModal, 5000); }, 100);
    });

    if (window.location.hash === '#subscribe') setTimeout(() => openSubscribeModal(), 300);
    window.addEventListener('hashchange', function() { if (window.location.hash === '#subscribe') openSubscribeModal(); });

    setTimeout(() => {
      document.querySelectorAll('#navbar-subscribe-btn, a[href*="#subscribe"], a.subscribe-link').forEach(link => {
        link.addEventListener('click', function(e) { e.preventDefault(); openSubscribeModal(); });
      });
    }, 1000);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initModal);
  else initModal();
})();
</script>

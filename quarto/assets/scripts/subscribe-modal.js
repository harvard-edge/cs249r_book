/**
 * Subscribe Modal Component
 * Elegant popup subscription form for ML Systems Textbook
 */

(function() {
  'use strict';

  // Create modal HTML structure
  function createModalHTML() {
    return `
      <div id="subscribe-modal" class="modal-overlay" style="display: none;">
        <div class="modal-container">
          <button class="modal-close" data-close-modal aria-label="Close">&times;</button>
          <div class="modal-content">
            <div class="modal-header">
              <div class="modal-icon">📚</div>
              <h2 class="modal-title">Get MLSysBook Updates</h2>
              <p class="modal-subtitle">Be the first to know about new chapters, labs, and resources.</p>
            </div>
            <form id="subscribe-modal-form" class="subscribe-form" action="https://buttondown.email/api/emails/embed-subscribe/mlsysbook" method="post">
              <div class="form-row">
                <div class="form-group">
                  <label for="modal-first-name">First name</label>
                  <input type="text" id="modal-first-name" name="metadata__first_name" required placeholder="First name">
                </div>
                <div class="form-group">
                  <label for="modal-last-name">Last name</label>
                  <input type="text" id="modal-last-name" name="metadata__last_name" required placeholder="Last name">
                </div>
              </div>
              <div class="form-group">
                <label for="modal-email">Email</label>
                <input type="email" id="modal-email" name="email" required placeholder="you@example.com">
              </div>
              <div class="form-group">
                <label>Role</label>
                <div class="role-options role-options-three-compact">
                  <label class="role-option">
                    <input type="radio" name="metadata__role" value="educator" required>
                    <span class="role-label">Educator</span>
                  </label>
                  <label class="role-option">
                    <input type="radio" name="metadata__role" value="student">
                    <span class="role-label">Student</span>
                  </label>
                  <label class="role-option">
                    <input type="radio" name="metadata__role" value="industry">
                    <span class="role-label">Industry</span>
                  </label>
                </div>
              </div>
              <div class="form-group">
                <label for="modal-organization">Organization <span class="optional-label">(optional)</span></label>
                <input type="text" id="modal-organization" name="metadata__organization" placeholder="University, company, or school">
              </div>
              <input type="hidden" name="tag" value="mlsysbook-textbook">
              <button type="submit" class="btn btn-primary subscribe-btn">Subscribe</button>
              <p class="form-note">No spam. Unsubscribe anytime.</p>
            </form>
            <div id="modal-subscribe-success" class="subscribe-success" style="display: none;">
              <div class="success-icon">🎉</div>
              <h3>You're subscribed!</h3>
              <p>Thanks for signing up. We'll keep you updated on new chapters, labs, and resources.</p>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Create modal CSS
  function createModalCSS() {
    const style = document.createElement('style');
    style.textContent = `
      /* Modal Overlay and Container - matching ReOrg style */
      .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(4px);
        z-index: 1000;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        animation: fadeIn 0.2s ease;
      }

      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      @keyframes slideUp {
        from { 
          opacity: 0;
          transform: translateY(20px) scale(0.98);
        }
        to { 
          opacity: 1;
          transform: translateY(0) scale(1);
        }
      }

      .modal-container {
        background: white;
        border-radius: 20px;
        max-width: 500px;
        width: 100%;
        max-height: 90vh;
        overflow-y: auto;
        position: relative;
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1), 0 0 0 1px rgba(0,0,0,0.05);
        animation: slideUp 0.3s ease;
        margin: auto;
      }

      .modal-close {
        position: absolute;
        top: 1rem;
        right: 1rem;
        width: 36px;
        height: 36px;
        border: none;
        background: #f8fafc;
        border-radius: 50%;
        font-size: 1.5rem;
        color: #64748b;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        z-index: 10;
        line-height: 1;
      }

      .modal-close:hover {
        background: white;
        color: #0f172a;
        transform: scale(1.05);
      }

      .modal-content {
        padding: 2.5rem;
      }

      .modal-header {
        text-align: center;
        margin-bottom: 2rem;
      }

      .modal-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
      }

      .modal-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
      }

      .modal-subtitle {
        font-size: 1rem;
        color: #475569;
        margin: 0;
      }

      /* Form Styles */
      .subscribe-form {
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
      }

      .form-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
      }

      .form-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
      }

      .form-group label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0f172a;
      }

      .optional-label {
        font-weight: 400;
        color: #64748b;
      }

      .form-group input[type="text"],
      .form-group input[type="email"] {
        padding: 0.875rem 1rem;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-size: 1rem;
        transition: all 0.2s ease;
        background: #f8fafc;
        font-family: inherit;
      }

      .form-group input[type="text"]:focus,
      .form-group input[type="email"]:focus {
        outline: none;
        border-color: #3b82f6;
        background: white;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
      }

      .form-group input::placeholder {
        color: #94a3b8;
      }

      /* Role Options - compact style like ReOrg */
      .role-options {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
      }

      .role-options-three-compact {
        grid-template-columns: repeat(3, 1fr);
      }

      .role-option {
        cursor: pointer;
      }

      .role-option input[type="radio"] {
        position: absolute;
        opacity: 0;
        width: 0;
        height: 0;
      }

      .role-label {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #475569;
        transition: all 0.2s ease;
        background: #f8fafc;
      }

      .role-options-three-compact .role-label {
        padding: 0.625rem 0.5rem;
        font-size: 0.8rem;
        text-align: center;
      }

      .role-option input[type="radio"]:checked + .role-label {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.08);
        color: #3b82f6;
      }

      .role-option:hover .role-label {
        border-color: #cbd5e1;
        background: white;
      }

      /* Button Styles */
      .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        border: none;
        cursor: pointer;
        font-family: inherit;
      }

      .btn-primary {
        background: #3b82f6;
        color: white;
      }

      .btn-primary:hover {
        background: #1e3a8a;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
      }

      .subscribe-btn {
        width: 100%;
        padding: 1rem;
        font-size: 1rem;
        margin-top: 0.5rem;
      }

      .form-note {
        text-align: center;
        font-size: 0.85rem;
        color: #64748b;
        margin: 0;
      }

      /* Success Message */
      .subscribe-success {
        text-align: center;
        padding: 2rem 1rem;
      }

      .success-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
      }

      .subscribe-success h3 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
      }

      .subscribe-success p {
        color: #475569;
        font-size: 1rem;
      }

      /* Dark mode support */
      body.quarto-dark .modal-container {
        background: #1e293b;
      }

      body.quarto-dark .modal-close {
        background: #0f172a;
        color: #94a3b8;
      }

      body.quarto-dark .modal-close:hover {
        background: #334155;
        color: #f1f5f9;
      }

      body.quarto-dark .modal-title,
      body.quarto-dark .form-group label,
      body.quarto-dark .subscribe-success h3 {
        color: #f1f5f9;
      }

      body.quarto-dark .modal-subtitle,
      body.quarto-dark .subscribe-success p {
        color: #cbd5e1;
      }

      body.quarto-dark .form-group input[type="text"],
      body.quarto-dark .form-group input[type="email"] {
        background: #0f172a;
        border-color: #334155;
        color: #f1f5f9;
      }

      body.quarto-dark .role-label {
        background: #0f172a;
        border-color: #334155;
        color: #cbd5e1;
      }

      body.quarto-dark .role-option input[type="radio"]:checked + .role-label {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
      }

      /* Responsive */
      @media (max-width: 640px) {
        .modal-content {
          padding: 2rem 1.5rem;
        }

        .form-row {
          grid-template-columns: 1fr;
        }

        .role-options-three-compact {
          grid-template-columns: repeat(3, 1fr);
        }
      }
    `;
    return style;
  }

  // Initialize modal
  function initModal() {
    // Add CSS
    document.head.appendChild(createModalCSS());

    // Add HTML
    const modalDiv = document.createElement('div');
    modalDiv.innerHTML = createModalHTML();
    document.body.appendChild(modalDiv.firstElementChild);

    const modal = document.getElementById('subscribe-modal');
    const form = document.getElementById('subscribe-modal-form');
    const success = document.getElementById('modal-subscribe-success');

    // Open modal function
    window.openModal = function() {
      modal.style.display = 'flex';
      document.body.style.overflow = 'hidden';
      
      // Focus first input
      setTimeout(() => {
        const firstInput = document.getElementById('modal-first-name');
        if (firstInput) firstInput.focus();
      }, 100);
    };

    // Close modal function
    window.closeModal = function() {
      modal.style.display = 'none';
      document.body.style.overflow = '';
      
      // Reset form after closing
      setTimeout(() => {
        form.style.display = 'flex';
        form.reset();
        success.style.display = 'none';
      }, 300);
    };

    // Close on overlay click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        closeModal();
      }
    });

    // Close button click
    const closeBtn = modal.querySelector('[data-close-modal]');
    if (closeBtn) {
      closeBtn.addEventListener('click', closeModal);
    }

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.style.display === 'flex') {
        closeModal();
      }
    });

    // Handle form submission
    form.addEventListener('submit', function() {
      // Let the form submit to Buttondown
      setTimeout(() => {
        form.style.display = 'none';
        success.style.display = 'block';
        
        // Close modal after 5 seconds
        setTimeout(closeModal, 5000);
      }, 100);
    });

    // Intercept navbar subscribe link
    setTimeout(() => {
      const navSubscribeLink = document.querySelector('a[href*="buttondown.email/mlsysbook"]');
      if (navSubscribeLink) {
        navSubscribeLink.addEventListener('click', function(e) {
          e.preventDefault();
          openModal();
        });
      }
    }, 1000);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initModal);
  } else {
    initModal();
  }
})();


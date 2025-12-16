import { injectStyles } from './modules/styles.js';
import { renderLayout, updateNavState } from './modules/ui.js?v=2';
import { getSession } from './modules/state.js?v=2';
import { openModal, closeModal, handleToggle, handleAuth, handleLogout, setMode, verifySession } from './modules/auth.js?v=2';
import { openProfileModal, closeProfileModal, handleProfileUpdate, geocodeAndSetCoordinates } from './modules/profile.js';
import { setupCameraEvents } from './modules/camera.js';
import { getBasePath } from './modules/config.js';

(function() {
    // 1. Inject CSS
    injectStyles();

    // 2. Render Layout
    renderLayout();

    // 3. Verify Session (Async)
    verifySession();

    // 4. Add Event Listeners
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.getElementById('sidebar');
    const authBtn = document.getElementById('authBtn');
    const authOverlay = document.getElementById('authOverlay');
    const authClose = document.getElementById('authClose');
    const authForm = document.getElementById('authForm');
    const authToggle = document.getElementById('authToggle');
    const forgotLink = document.getElementById('authForgotLink');

    // Profile Modal Elements
    const profileOverlay = document.getElementById('profileOverlay');
    const profileClose = document.getElementById('profileClose');
    const profileForm = document.getElementById('profileForm');
    const profileLogoutBtn = document.getElementById('profileLogoutBtn');

    // Camera & Image Logic
    setupCameraEvents();

    // Wire up events
    if (menuBtn && sidebar) {
        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('active') &&
                !sidebar.contains(e.target) &&
                !menuBtn.contains(e.target)) {
                menuBtn.classList.remove('active');
                sidebar.classList.remove('active');
            }
        });

        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menuBtn.classList.toggle('active');
            sidebar.classList.toggle('active');
        });
    }

    if (authBtn) {
        authBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const { isLoggedIn } = getSession();
            if (isLoggedIn) {
                openProfileModal();
            } else {
                openModal();
            }
        });
    }

    if (authClose) authClose.addEventListener('click', closeModal);
    if (authOverlay) {
        // authOverlay.addEventListener('click', (e) => {
        //     if (e.target === authOverlay) closeModal();
        // });
    }
    if (authToggle) authToggle.addEventListener('click', handleToggle);
    if (forgotLink) forgotLink.addEventListener('click', () => setMode('forgot'));
    if (authForm) authForm.addEventListener('submit', handleAuth);

    // Profile Modal Events
    if (profileClose) profileClose.addEventListener('click', closeProfileModal);
    if (profileOverlay) {
        // profileOverlay.addEventListener('click', (e) => {
        //     if (e.target === profileOverlay) closeProfileModal();
        // });
    }
    if (profileLogoutBtn) profileLogoutBtn.addEventListener('click', handleLogout);
    if (profileForm) profileForm.addEventListener('submit', handleProfileUpdate);

    // Add blur event listener for geocoding the location
    const profileLocationInput = document.getElementById('profileLocation');
    if (profileLocationInput) {
        profileLocationInput.addEventListener('blur', () => {
            geocodeAndSetCoordinates(profileLocationInput.value);
        });
    }

    // Check for redirect action
    const params = new URLSearchParams(window.location.search);
    const action = params.get('action');

    if (action === 'login') {
        localStorage.removeItem("tinytorch_token");
        localStorage.removeItem("tinytorch_refresh_token");
        localStorage.removeItem("tinytorch_user");
        updateNavState();
        openModal('login');
    } else if (action === 'profile') {
        const { isLoggedIn } = getSession();
        if (isLoggedIn) {
            openProfileModal();
        } else {
            openModal('login');
        }
    } else if (action === 'join') {
        const { isLoggedIn } = getSession();
        if (!isLoggedIn) {
            openModal('signup');
        }
    } else if (params.get('community')) {
        window.location.href = getBasePath() + '/community.html';
        return;
    }
})();

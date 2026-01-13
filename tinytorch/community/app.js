import { injectStyles } from './modules/styles.js';
import { renderLayout, updateNavState } from './modules/ui.js?v=3';
import { getSession } from './modules/state.js?v=2';
import { openModal, closeModal, handleToggle, handleAuth, handleLogout, setMode, verifySession, signInWithSocial, supabase } from './modules/auth.js?v=3';
import { openProfileModal, closeProfileModal, handleProfileUpdate, geocodeAndSetCoordinates, checkAndAutoUpdateLocation } from './modules/profile.js';
import { setupCameraEvents } from './modules/camera.js';
import { getBasePath } from './modules/config.js';

(function() {
    // 1. Inject CSS
    injectStyles();

    // 2. Render Layout
    renderLayout();

    // 2.5 Attach Social Login Listeners
    const btnGoogle = document.getElementById('btn-login-google');
    const btnGithub = document.getElementById('btn-login-github');
    
    if (btnGoogle) {
        btnGoogle.addEventListener('click', (e) => { 
            e.preventDefault(); 
            signInWithSocial('google'); 
            closeModal();
        });
    }
    if (btnGithub) {
        btnGithub.addEventListener('click', (e) => { 
            e.preventDefault(); 
            signInWithSocial('github'); 
            closeModal();
        });
    }

    // 2.6 Check for Supabase Session & Verify
    const checkProfile = async (session) => {
        if (!session || window.location.pathname.includes('profile_setup')) return;
        
        const { data: profile } = await supabase
            .from('profiles')
            .select('display_name, institution, location')
            .eq('id', session.user.id)
            .single();

        const hasName = profile && profile.display_name;
        const hasInst = profile && profile.institution && (Array.isArray(profile.institution) ? profile.institution.length > 0 : !!profile.institution);
        const hasLoc = profile && profile.location;

        if (!hasName || !hasInst || !hasLoc) {
            window.location.href = getBasePath() + '/profile_setup.html';
        }
    };

    supabase.auth.getSession().then(({ data: { session } }) => {
        if (session) {
             localStorage.setItem("tinytorch_token", session.access_token);
             if (session.refresh_token) localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
             if (session.user) localStorage.setItem("tinytorch_user", JSON.stringify(session.user));
             
             // Clean URL hash if present (Supabase puts tokens there)
             if (window.location.hash && window.location.hash.includes('access_token')) {
                 window.history.replaceState({}, document.title, window.location.pathname + window.location.search);
             }
             
             updateNavState();
             checkProfile(session);
        }
        // 3. Verify Session (Async)
        verifySession();
        
        // 3.5 Check and Auto-update Location (Async, non-blocking)
        checkAndAutoUpdateLocation();
    });

    // 2.7 Listen for Auth Changes (Fixes OAuth redirect lag)
    supabase.auth.onAuthStateChange((event, session) => {
        if (event === 'SIGNED_IN' && session) {
            localStorage.setItem("tinytorch_token", session.access_token);
            if (session.refresh_token) localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
            if (session.user) localStorage.setItem("tinytorch_user", JSON.stringify(session.user));
            updateNavState();
            checkProfile(session);
        } else if (event === 'SIGNED_OUT') {
            localStorage.removeItem("tinytorch_token");
            updateNavState();
        }
    });

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

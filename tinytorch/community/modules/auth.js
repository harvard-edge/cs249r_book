import { NETLIFY_URL, SUPABASE_URL, SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY, getBasePath } from './config.js';
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';
import { updateNavState } from './ui.js?v=2';
import { closeProfileModal, openProfileModal } from './profile.js';
import { getSession, forceLogin, clearSession } from './state.js?v=2';

// Initialize Supabase Client
const supabase = createClient(SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY);
export { supabase };

export async function signInWithSocial(provider) {
    const isIframe = window.self !== window.top;
    const basePath = getBasePath();
    const redirectTo = window.location.origin + basePath + (isIframe ? '/auth_callback.html' : '/index.html');

    if (isIframe) {
        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: provider,
            options: {
                redirectTo: redirectTo,
                skipBrowserRedirect: true
            }
        });
        if (error) {
            console.error('Social login error:', error);
            alert('Login failed: ' + error.message);
            return;
        }
        if (data && data.url) {
            const width = 600, height = 700;
            const left = (window.screen.width - width) / 2;
            const top = (window.screen.height - height) / 2;
            window.open(data.url, 'tinytorch_auth', `width=${width},height=${height},left=${left},top=${top}`);
        }
    } else {
        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: provider,
            options: {
                redirectTo: redirectTo
            }
        });
        if (error) {
            console.error('Social login error:', error);
            alert('Login failed: ' + error.message);
        }
    }
}

// Listen for popup auth messages
window.addEventListener('message', async (event) => {
    if (event.origin !== window.location.origin) return;

    if (event.data && event.data.type === 'TINY_TORCH_AUTH_SUCCESS') {
        const session = event.data.session;
        if (session) {
            console.log("Auth success message received in iframe, syncing session...");
            const { error } = await supabase.auth.setSession({
                access_token: session.access_token,
                refresh_token: session.refresh_token
            });
            if (error) {
                console.error("Error setting session from popup:", error);
            } else {
                // If we are on community page, just close the modal.
                // onAuthStateChange in app.js will handle the rest.
                if (window.location.pathname.includes('community.html')) {
                    closeModal();
                }
            }
        }
    }
});

let currentMode = 'signup';

export async function refreshToken() {
    const refreshToken = localStorage.getItem("tinytorch_refresh_token");
    if (!refreshToken) {
        return false;
    }

    try {
        const { data, error } = await supabase.auth.refreshSession({ refresh_token: refreshToken });

        if (error || !data.session) {
            console.error("Supabase refresh failed:", error);
            return false;
        }

        const session = data.session;
        localStorage.setItem("tinytorch_token", session.access_token);
        if (session.refresh_token) {
            localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
        }
        if (session.user) {
            localStorage.setItem("tinytorch_user", JSON.stringify(session.user));
        }
        return session.access_token;
    } catch (e) {
        console.error("Token refresh failed", e);
    }
    return false;
}

export async function verifySession() {
    const { token } = getSession();
    const refreshTokenStr = localStorage.getItem("tinytorch_refresh_token");
    
    if (!token) return;

    // 1. Ensure Supabase Client is synced
    // If the page reloaded, the Supabase client might not have initialized the session 
    // from its own storage yet, or persistence might have failed.
    // We check if it knows about the session.
    const { data: { session: sbSession } } = await supabase.auth.getSession();
    
    // If Supabase client is empty but we have tokens (from Direct Email login), restore it.
    if (!sbSession && refreshTokenStr) {
        const { error } = await supabase.auth.setSession({
            access_token: token,
            refresh_token: refreshTokenStr
        });
        if (error) {
            console.warn("Auto-sync setSession failed:", error);
        }
    }

    try {
        // Use a lightweight call to check validity. 
        // We use get-profile-details but we could also just decode if we trusted client time.
        // A network call is safer.
        const response = await fetch(`${SUPABASE_URL}/get-profile-details`, { 
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            return; // Token is valid
        }

        if (response.status === 401 || response.status === 403) {
            console.log("Session verification failed. Attempting refresh...");
            const newToken = await refreshToken();
            if (newToken) {
                console.log("Session refreshed successfully.");
                updateNavState(); // Update UI with new state if needed
            } else {
                console.warn("Session refresh failed. Clearing session.");
                clearSession();
                updateNavState();
            }
        }
    } catch (e) {
        console.error("Session verification error", e);
    }
}

export function setMode(mode) {
    const emailInput = document.getElementById('authEmail');
    const passwordInput = document.getElementById('authPassword');
    const authTitle = document.getElementById('authTitle');
    const authSubmit = document.getElementById('authSubmit');
    const authToggle = document.getElementById('authToggle');
    const forgotContainer = document.getElementById('authForgotContainer');
    const authError = document.getElementById('authError');
    const authForm = document.getElementById('authForm');

    if (!emailInput) return; // Guard if DOM not ready

    const previousEmail = emailInput.value;
    currentMode = mode;

    authForm.reset();
    authError.style.display = 'none';
    authError.textContent = '';
    if (mode === 'forgot') {
        emailInput.value = '';
    } else {
         emailInput.value = previousEmail;
    }

    if (mode === 'login') {
        authTitle.textContent = 'Login';

        // Check for confirmed_email param
        const params = new URLSearchParams(window.location.search);
        if (params.get('confirmed_email') === 'true') {
             authTitle.textContent = 'Thank you for confirming your email. Please login.';
             // Clean up URL
             const newUrl = window.location.pathname + '?action=login';
             window.history.replaceState({}, '', newUrl);
        }

        authSubmit.textContent = 'Login';
        authToggle.textContent = 'Need an account? Create Account';
        passwordInput.classList.remove('hidden');
        passwordInput.required = true;
        forgotContainer.classList.remove('hidden');
    } else if (mode === 'signup') {
        authTitle.textContent = 'Create Account';
        authSubmit.textContent = 'Create Account';
        authToggle.textContent = 'Already have an account? Login';
        passwordInput.classList.remove('hidden');
        passwordInput.required = true;
        forgotContainer.classList.add('hidden');
    } else if (mode === 'forgot') {
        authTitle.textContent = 'Reset Password';
        authSubmit.textContent = 'Send Reset Link';
        authToggle.textContent = 'Back to Login';
        passwordInput.classList.add('hidden');
        passwordInput.required = false;
        forgotContainer.classList.add('hidden');
    }
}

export function handleToggle() {
    if (currentMode === 'login') {
        setMode('signup');
    } else if (currentMode === 'signup') {
        setMode('login');
    } else if (currentMode === 'forgot') {
        setMode('login');
    }
}

export function showMessageModal(title, body, onCloseCallback = null) {
    const overlay = document.getElementById('messageOverlay');
    const titleEl = document.getElementById('messageTitle');
    const bodyEl = document.getElementById('messageBody');
    const btnEl = document.getElementById('messageBtn');
    
    if (overlay && titleEl && bodyEl) {
        titleEl.textContent = title;
        bodyEl.textContent = body;
        overlay.classList.add('active');
        
        // Remove previous listeners to avoid duplicates if reused
        const newBtn = btnEl.cloneNode(true);
        btnEl.parentNode.replaceChild(newBtn, btnEl);
        
        newBtn.addEventListener('click', () => {
            overlay.classList.remove('active');
            if (onCloseCallback) onCloseCallback();
        });
    }
}

export async function handleAuth(e) {
    e.preventDefault();
    const emailInput = document.getElementById('authEmail');
    const passwordInput = document.getElementById('authPassword');
    const authError = document.getElementById('authError');
    const authSubmit = document.getElementById('authSubmit');
    const basePath = getBasePath();

    const email = emailInput.value;
    const password = passwordInput.value;
    authError.style.display = 'none';
    
    // Save original text
    const originalBtnText = authSubmit.textContent;
    // Show spinner
    authSubmit.disabled = true;
    authSubmit.innerHTML = '<div class="spinner"></div>';

    try {
        let endpoint, body;

        if (currentMode === 'forgot') {
            endpoint = '/api/auth/reset-password';
            body = { email };
        } else {
            endpoint = currentMode === 'login' ? '/api/auth/login' : '/api/auth/signup';
            body = {
                email,
                password,
                redirect_to: window.location.origin + basePath + '/index.html?action=login&confirmed_email=true'
            };
        }

        const url = `${NETLIFY_URL}${endpoint}`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body),
            credentials: 'include'
        });

        const data = await response.json();

        if (currentMode === 'forgot') {
                if (response.ok) {
                    showMessageModal('Reset Link Sent', data.message || 'If an account exists, a reset link has been sent.');
                    setMode('login');
                } else {
                    throw new Error(data.error || 'Failed to send reset link');
                }
        } else {
            if (!response.ok) {
                throw new Error(data.error || (currentMode === 'login' ? 'Login failed' : 'Signup failed'));
            }

            if (currentMode === 'login') {
                if (data.access_token) {
                    localStorage.setItem("tinytorch_token", data.access_token);
                    if (data.refresh_token) localStorage.setItem("tinytorch_refresh_token", data.refresh_token);
                    localStorage.setItem("tinytorch_user", JSON.stringify(data.user));

                    // Sync Supabase Client so it doesn't trigger SIGNED_OUT
                    if (data.refresh_token) {
                        const { error: sessionError } = await supabase.auth.setSession({
                            access_token: data.access_token,
                            refresh_token: data.refresh_token
                        });
                        if (sessionError) {
                            console.error("Supabase setSession error during login:", sessionError);
                        }
                    }

                    updateNavState();

                    // Check Profile Completeness immediately
                    const { data: profile } = await supabase
                        .from('profiles')
                        .select('display_name, institution, location')
                        .eq('id', data.user.id)
                        .single();

                    const hasName = profile && profile.display_name;
                    const hasInst = profile && profile.institution && (Array.isArray(profile.institution) ? profile.institution.length > 0 : !!profile.institution);
                    const hasLoc = profile && profile.location;

                    if (!hasName || !hasInst || !hasLoc) {
                        window.location.href = basePath + '/profile_setup.html';
                        return;
                    }

                    const params = new URLSearchParams(window.location.search);
                    if (params.get('action') === 'profile') {
                        closeModal();
                        openProfileModal();
                    } else {
                        window.location.href = basePath + '/dashboard.html';
                    }
                }
            } else {
                // Signup Success - Show Message Modal
                // We close the auth modal first so it doesn't overlap
                closeModal();
                showMessageModal(
                    'Check your Email', 
                    'If you don\'t already have an account, we have sent you an email. Please check your inbox to confirm your signup.',
                    () => {
                        window.location.href = basePath + '/dashboard.html';
                    }
                );
            }
        }

    } catch (error) {
        console.error("Auth error:", error);
        authError.textContent = error.message;
        authError.style.display = 'block';
    } finally {
        authSubmit.disabled = false;
        // Restore button text based on current mode, as logic might have changed mode or just finished
        if (currentMode === 'login') authSubmit.textContent = 'Login';
        else if (currentMode === 'signup') authSubmit.textContent = 'Create Account';
        else authSubmit.textContent = 'Send Reset Link';
    }
}

export async function handleLogout() {
    const basePath = getBasePath();
    if (confirm('Are you sure you want to logout?')) {
        await supabase.auth.signOut();
        localStorage.removeItem("tinytorch_token");
        localStorage.removeItem("tinytorch_refresh_token");
        localStorage.removeItem("tinytorch_user");
        sessionStorage.removeItem("tinytorch_location_checked");
        updateNavState();
        closeProfileModal();
        window.location.href = basePath + '/index.html';
    }
}

export function openModal(mode = 'signup') {
    const authOverlay = document.getElementById('authOverlay');
    authOverlay.classList.add('active');
    setMode(mode);
}

export function closeModal() {
    const authOverlay = document.getElementById('authOverlay');
    authOverlay.classList.remove('active');
}

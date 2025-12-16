import { NETLIFY_URL, SUPABASE_URL, getBasePath } from './config.js';
import { updateNavState } from './ui.js?v=2';
import { closeProfileModal, openProfileModal } from './profile.js';
import { getSession, forceLogin, clearSession } from './state.js?v=2';

let currentMode = 'signup';

export async function refreshToken() {
    const refreshToken = localStorage.getItem("tinytorch_refresh_token");
    if (!refreshToken) {
        return false;
    }

    try {
        const refreshRes = await fetch(`${NETLIFY_URL}/api/auth/refresh`, { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refreshToken })
        });

        if (!refreshRes.ok) { 
            return false; 
        }

        const refreshData = await refreshRes.json();
        const session = refreshData.session || refreshData; 

        if (session && session.access_token) {
            localStorage.setItem("tinytorch_token", session.access_token);
            if (session.refresh_token) {
                localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
            }
            if (session.user) {
                localStorage.setItem("tinytorch_user", JSON.stringify(session.user));
            }
            return session.access_token;
        }
    } catch (e) {
        console.error("Token refresh failed", e);
    }
    return false;
}

export async function verifySession() {
    const { token } = getSession();
    if (!token) return;

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
    authSubmit.disabled = true;
    authSubmit.textContent = 'Processing...';

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
                    alert(data.message || 'If an account exists, a reset link has been sent.');
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
                    updateNavState();

                    const params = new URLSearchParams(window.location.search);
                    if (params.get('action') === 'profile') {
                        closeModal();
                        openProfileModal();
                    } else {
                        window.location.href = basePath + '/dashboard.html';
                    }
                }
            } else {
                alert('If an account exists for this email, we have sent you a login link. Otherwise, please check your email to confirm your signup.');
                window.location.href = basePath + '/dashboard.html';
            }
        }

    } catch (error) {
        console.error("Auth error:", error);
        authError.textContent = error.message;
        authError.style.display = 'block';
    } finally {
        authSubmit.disabled = false;
        if (currentMode === 'login') authSubmit.textContent = 'Login';
        else if (currentMode === 'signup') authSubmit.textContent = 'Create Account';
        else authSubmit.textContent = 'Send Reset Link';
    }
}

export function handleLogout() {
    const basePath = getBasePath();
    if (confirm('Are you sure you want to logout?')) {
        localStorage.removeItem("tinytorch_token");
        localStorage.removeItem("tinytorch_user");
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

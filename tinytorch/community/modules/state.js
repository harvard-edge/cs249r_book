import { getBasePath } from './config.js';

// State Management
export function getSession() {
    const token = localStorage.getItem("tinytorch_token");
    const userStr = localStorage.getItem("tinytorch_user");
    let user = null;
    try { user = JSON.parse(userStr); } catch(e) {}

    // Allow window.USER_EMAIL to override or serve as fallback if no local storage (legacy support)
    const email = user ? user.email : (window.USER_EMAIL || null);
    return { token, email, isLoggedIn: !!token };
}

export function clearSession() {
    localStorage.removeItem("tinytorch_token");
    localStorage.removeItem("tinytorch_refresh_token");
    localStorage.removeItem("tinytorch_user");
}

export function forceLogin() {
    console.warn("Session expired or invalid. Redirecting to login...");
    clearSession();
    window.location.href = getBasePath() + '/index.html?action=login';
}

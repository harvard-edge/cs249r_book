import { getBasePath } from './config.js';
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';
import { SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY } from './config.js';

const supabase = createClient(SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY, {
    auth: {
        flowType: 'pkce', // Prefer PKCE for security, or keep 'implicit' if standard for this app
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true
    }
});
export { supabase };

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

export async function clearSession() {
    console.log("🧹 Aggressively clearing session and cookies...");
    try {
        await supabase.auth.signOut();
    } catch (e) {
        console.warn("Supabase signOut error during clearSession:", e);
    }
    
    // 1. Explicitly remove our own keys
    localStorage.removeItem("tinytorch_token");
    localStorage.removeItem("tinytorch_refresh_token");
    localStorage.removeItem("tinytorch_user");
    sessionStorage.removeItem("tinytorch_location_checked");

    // 2. Clear all Supabase and auth-related keys from localStorage
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (
            key.includes("supabase") || 
            key.includes("auth-token") || 
            key.startsWith("sb-")
        )) {
            keysToRemove.push(key);
        }
    }
    keysToRemove.forEach(k => localStorage.removeItem(k));

    // 3. Clear all auth-related cookies
    const cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i];
        const eqPos = cookie.indexOf("=");
        const name = eqPos > -1 ? cookie.substr(0, eqPos).trim() : cookie.trim();
        // Clear for all common paths and subdomains
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/;domain=" + window.location.hostname;
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
    }
    
    console.log("✨ Session cleared.");
}

export function forceLogin() {
    console.warn("Session expired or invalid. Redirecting to login...");
    clearSession();
    window.location.href = getBasePath() + '/index.html?action=login';
}

import { SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY, SUPABASE_URL, getBasePath } from './config.js';
import { clearSession, supabase } from './state.js';

const LOGIN_PAGE = getBasePath() + '/index.html';
const SETUP_PAGE = getBasePath() + '/profile_setup.html';
const DASHBOARD_PAGE = getBasePath() + '/dashboard.html';

(async function guard() {
  const path = window.location.pathname;
  // Enhanced path checking for landing page
  const isOnIndex = path.endsWith('/') || path.endsWith('index.html') || path === getBasePath() || path === getBasePath() + '/';
  const isOnSetupPage = path.includes('profile_setup.html');
  const isPublicPage = isOnIndex || path.includes('login') || path.includes('about') || path.includes('contact');
  const isProtected = !isPublicPage;

  // 0. Get Session
  const storedToken = localStorage.getItem("tinytorch_token");
  const storedRefresh = localStorage.getItem("tinytorch_refresh_token");
  
  if (storedToken && storedRefresh) {
      try {
          await supabase.auth.setSession({
              access_token: storedToken,
              refresh_token: storedRefresh
          });
      } catch (e) {
          console.warn("Guard: setSession failed", e);
      }
  }

  const { data: { session } } = await supabase.auth.getSession();
  
  if (!session && isProtected) {
      console.log("🚧 No session on protected page. Redirecting to login...");
      window.location.href = LOGIN_PAGE + '?action=login&next=' + encodeURIComponent(path);
      return;
  }

  if (!session) return; // Public page, no session, we are fine.

  // 1. Fetch Profile with Timeout
  let profile = null;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
      // Use the API for profile details
      const res = await fetch(`${SUPABASE_URL}/get-profile-details`, {
          headers: { 'Authorization': `Bearer ${session.access_token}` },
          signal: controller.signal
      });
      clearTimeout(timeoutId);
      
      if (res.ok) {
          const data = await res.json();
          profile = data.profile;
      } else if (res.status === 401 || res.status === 404) {
          // 401: Token expired, 404: User deleted from DB
          console.warn(`Guard: Session invalid or account deleted (${res.status}). Purging...`);
          await clearSession();
          
          if (isProtected) {
              window.location.href = LOGIN_PAGE + '?action=login';
          } else {
              // If on a public page, just reload to clear UI state
              window.location.reload();
          }
          return;
      }
  } catch (e) {
      console.warn("Guard: Profile fetch failed or timed out", e);
  }

  // 2. The Rules
  const hasName = profile && profile.display_name;
  const hasInst = profile && profile.institution && (Array.isArray(profile.institution) ? profile.institution.length > 0 : !!profile.institution);
  const hasLoc = profile && profile.location;

  const isComplete = hasName && hasInst && hasLoc;

  if (isComplete) {
      if (isOnSetupPage) {
          console.log("✅ Profile complete. Moving to dashboard...");
          window.location.href = DASHBOARD_PAGE;
      }
  } else {
      if (!isOnSetupPage && isProtected) {
          console.log("🚧 Profile incomplete. Redirecting to setup...");
          window.location.href = SETUP_PAGE;
      }
  }
})();

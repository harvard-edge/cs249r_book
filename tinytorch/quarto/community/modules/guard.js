import { SUPABASE_URL, NETLIFY_URL, getBasePath } from './config.js';
import { clearSession } from './state.js';

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

  const redirectToLogin = () => {
      window.location.href = LOGIN_PAGE + '?action=login&next=' + encodeURIComponent(path);
  };

  // 0. The app authenticates with `tinytorch_token` (set by email login and OAuth alike).
  //    Use it as the source of truth here too — older email logins never establish a
  //    Supabase client session, so gating on supabase.auth.getSession() locked them out.
  let token = localStorage.getItem("tinytorch_token");

  if (!token) {
      if (isProtected) redirectToLogin();
      return; // Public page with no token: nothing to guard.
  }

  // 1. Validate the token against the profile Edge Function, with one refresh retry.
  let profile = null;
  let retryCount = 0;
  const MAX_RETRIES = 1;

  do {
      let res;
      try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 5000);
          res = await fetch(`${SUPABASE_URL}/get-profile-details`, {
              headers: { 'Authorization': `Bearer ${token}` },
              signal: controller.signal
          });
          clearTimeout(timeoutId);
      } catch (e) {
          // Network error / timeout: fail open rather than bounce the user around.
          console.warn("Guard: profile fetch failed or timed out", e);
          return;
      }

      // 401, or 400 with "Invalid Token", means the access token needs refreshing.
      let needsRefresh = res.status === 401;
      if (res.status === 400) {
          try {
              const errData = await res.clone().json();
              if (errData?.error?.includes("Invalid Token")) needsRefresh = true;
          } catch (e) { /* ignore parse error */ }
      }

      if (needsRefresh && retryCount === 0) {
          const refreshTokenStr = localStorage.getItem("tinytorch_refresh_token");
          if (refreshTokenStr) {
              try {
                  const refreshRes = await fetch(`${NETLIFY_URL}/api/auth/refresh`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ refreshToken: refreshTokenStr })
                  });
                  if (refreshRes.ok) {
                      const refreshData = await refreshRes.json();
                      const session = refreshData.session || refreshData;
                      if (session?.access_token) {
                          token = session.access_token;
                          localStorage.setItem("tinytorch_token", token);
                          if (session.refresh_token) {
                              localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
                          }
                          retryCount++;
                          continue; // Retry with refreshed token
                      }
                  }
              } catch (e) {
                  console.warn("Guard: token refresh failed", e);
              }
          }
          // No refresh token, or refresh failed → the session is dead.
          await clearSession();
          if (isProtected) redirectToLogin();
          return;
      }

      if (res.status === 404) {
          // Account deleted server-side.
          console.warn("Guard: account not found (404). Purging session...");
          await clearSession();
          if (isProtected) redirectToLogin();
          return;
      }

      if (!res.ok) {
          // Unknown server error: fail open rather than lock the user out.
          console.warn("Guard: unexpected profile fetch status", res.status);
          return;
      }

      try {
          const data = await res.json();
          profile = data.profile;
      } catch (e) {
          console.warn("Guard: failed to parse profile response", e);
          return;
      }
      break; // Success
  } while (retryCount <= MAX_RETRIES);

  // 2. Completeness rules.
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

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';
import { SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY, SUPABASE_URL, getBasePath } from './config.js';

const supabase = createClient(SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY);
const LOGIN_PAGE = getBasePath() + '/index.html';
const SETUP_PAGE = getBasePath() + '/profile_setup.html';

(async function guard() {
  // 0. HYDRATE SESSION (Fix for Direct Email Login)
  const storedToken = localStorage.getItem("tinytorch_token");
  const storedRefresh = localStorage.getItem("tinytorch_refresh_token");
  if (storedToken && storedRefresh) {
      await supabase.auth.setSession({
          access_token: storedToken,
          refresh_token: storedRefresh
      });
  }

  // 1. Check Session (Supabase Client)
  const { data: { session } } = await supabase.auth.getSession();
  
  let profile = null;

  if (session) {
      // 2a. Fetch Profile via Client
      const { data } = await supabase
        .from('profiles')
        .select('display_name, institution, location')
        .eq('id', session.user.id)
        .single();
      profile = data;
  } else {
      // 1b. Fallback: Check Token Manually
      if (!storedToken) {
          // No session, no token -> Redirect
          if (!window.location.pathname.includes('index') && !window.location.pathname.includes('login') && !window.location.pathname.includes('about')) {
              window.location.href = LOGIN_PAGE + '?action=login&next=' + encodeURIComponent(window.location.pathname);
          }
          return;
      }

      // Have token, verify via API
      try {
          const res = await fetch(`${SUPABASE_URL}/get-profile-details`, {
              headers: { 'Authorization': `Bearer ${storedToken}` }
          });
          
          if (!res.ok) {
              throw new Error("Token invalid");
          }
          const data = await res.json();
          profile = data.profile; // API returns { profile: {...}, completed_modules: [...] }
          
      } catch (e) {
          console.warn("Guard: Token validation failed", e);
          if (!window.location.pathname.includes('index') && !window.location.pathname.includes('login')) {
              window.location.href = LOGIN_PAGE + '?action=login&next=' + encodeURIComponent(window.location.pathname);
          }
          return;
      }
  }

  // 3. The Rules
  // Must have ALL three: Name, Institution, Location
  const hasName = profile && profile.display_name;
  const hasInst = profile && profile.institution && (Array.isArray(profile.institution) ? profile.institution.length > 0 : !!profile.institution);
  const hasLoc = profile && profile.location;

  const isComplete = hasName && hasInst && hasLoc;
  const isOnSetupPage = window.location.pathname.includes('profile_setup');

  if (!isComplete && !isOnSetupPage) {
     console.log("🚧 Profile incomplete. Redirecting to setup...");
     window.location.href = SETUP_PAGE;
  } 
  else if (isComplete && isOnSetupPage) {
     // If they are done but try to visit setup, send them to dashboard
     window.location.href = getBasePath() + '/dashboard.html';
  }

})();

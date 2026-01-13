import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';
import { SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY, getBasePath } from './config.js';

const supabase = createClient(SUPABASE_PROJECT_URL, SUPABASE_ANON_KEY);
const LOGIN_PAGE = getBasePath() + '/index.html';
const SETUP_PAGE = getBasePath() + '/profile_setup.html';

(async function guard() {
  // 1. Check Session
  const { data: { session } } = await supabase.auth.getSession();

  if (!session) {
    // If not logged in and not on login page, kick them out
    if (!window.location.pathname.includes('index') && !window.location.pathname.includes('login') && !window.location.pathname.includes('about')) {
        // Double check against localStorage to avoid flicker if possible, but session check is authoritative
        window.location.href = LOGIN_PAGE + '?action=login&next=' + encodeURIComponent(window.location.pathname);
    }
    return;
  }

  // 2. Check Profile Completeness
  // We fetch only the required fields to be fast
  const { data: profile } = await supabase
    .from('profiles')
    .select('display_name, institution, location')
    .eq('id', session.user.id)
    .single();

  // 3. The Rules
  // Must have ALL three: Name, Institution, Location
  // Note: institution is often stored as array in this DB based on other files, but sometimes string.
  // Let's check truthiness.
  
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

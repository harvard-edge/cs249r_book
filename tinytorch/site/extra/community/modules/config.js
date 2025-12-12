// Configuration
export const SUPABASE_URL = "https://zrvmjrxhokwwmjacyhpq.supabase.co/functions/v1";
export const NETLIFY_URL = "https://tinytorch.netlify.app"; 

// URL Base Path Logic for Community Site Hosting
export function getBasePath() {
    let basePath = '';
    const hostname = window.location.hostname;
    
    if (hostname === 'mlsysbook.ai') {
        basePath = '/tinytorch/community';
    } else if (hostname === 'tinytorch.ai' || (hostname === 'localhost' && window.location.port === '8000')) {
        basePath = '/community';
    } else if (hostname === 'harvard-edge.github.io') {
        basePath = '/cs249r_book_dev/tinytorch/community';
    }
    return basePath;
}

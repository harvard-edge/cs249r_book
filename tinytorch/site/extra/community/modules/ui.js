import { getBasePath, NETLIFY_URL } from './config.js';
import { getSession } from './state.js?v=2';

export function updateNavState() {
    const { isLoggedIn, email: userEmail } = getSession();

    const logoutIcon = `<path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>`;
    const userIcon = `<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>`;

    const currentIcon = isLoggedIn ? logoutIcon : userIcon;
    const btnText = isLoggedIn ? `${userEmail}` : 'Create or Login into Account';
    const btnClass = isLoggedIn ? 'login-btn logged-in' : 'login-btn';

    const authBtnElement = document.getElementById('authBtn');
    const navDashboard = document.getElementById('navDashboardBtn');
    const navCommunity = document.getElementById('navCommunityBtn');
    const navArena = document.getElementById('navArenaBtn');

    if (authBtnElement) {
        authBtnElement.className = btnClass;
        authBtnElement.querySelector('.login-icon').innerHTML = currentIcon;
        authBtnElement.querySelector('.btn-text').textContent = btnText;
    }

    // Show/Hide Extra Icons
    if (navDashboard) navDashboard.style.display = isLoggedIn ? 'flex' : 'none';
    if (navCommunity) navCommunity.style.display = 'flex';
    if (navArena) navArena.style.display = 'flex';
}

export function renderLayout() {
    const basePath = getBasePath();
    const { isLoggedIn, email: userEmail } = getSession();

    const logoutIcon = `<path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>`;
    const userIcon = `<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>`;

    const footprintsIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-footprints-icon lucide-footprints"><path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 8.68V16a2 2 0 1 1-4 0Z"/><path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z"/><path d="M16 17h4"/><path d="M4 13h4"/></svg>`;
    const globeIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-earth-icon lucide-earth"><path d="M21.54 15H17a2 2 0 0 0-2 2v4.54"/><path d="M7 3.34V5a3 3 0 0 0 3 3a2 2 0 0 1 2 2c0 1.1.9 2 2 2a2 2 0 0 0 2-2c0-1.1.9-2 2-2h3.17"/><path d="M11 21.95V18a2 2 0 0 0-2-2a2 2 0 0 1-2-2v-1a2 2 0 0 0-2-2H2.05"/><circle cx="12" cy="12" r="10"/></svg>`;
    const arenaIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-layout-grid"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg>`;

    const currentIcon = isLoggedIn ? logoutIcon : userIcon;
    const btnText = isLoggedIn ? `Logout of ${userEmail}` : 'Create Account';
    const btnClass = isLoggedIn ? 'login-btn logged-in' : 'login-btn';
    const displayExtras = isLoggedIn ? 'flex' : 'none';

    // Social Login Setup
    const googleIcon = `<svg class="social-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>`;
    const githubIcon = `<svg class="social-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>`;

    const layoutHTML = `
        <button class="menu-btn" id="menuBtn">
            <span></span>
            <span></span>
            <span></span>
        </button>

        <div class="top-right-actions">
            <a href="#" class="${btnClass}" id="authBtn">
                <svg class="login-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    ${currentIcon}
                </svg>
                <span class="btn-text">${btnText}</span>
            </a>

            <a href="${basePath}/dashboard.html" class="nav-icon-btn" id="navDashboardBtn" style="display: ${displayExtras};" title="Dashboard">
                ${footprintsIcon}
            </a>

            <a href="${basePath}/arena.html" class="nav-icon-btn" id="navArenaBtn" style="display: flex;" title="Arena">
                ${arenaIcon}
            </a>

            <a href="${basePath}/community.html" class="nav-icon-btn" id="navCommunityBtn" style="display: flex;" title="Community">
                ${globeIcon}
            </a>
        </div>

        <nav class="sidebar" id="sidebar">
            <a href="${basePath}/index.html" class="nav-item">Home</a>
            <a href="${basePath}/about.html" class="nav-item" id="aboutLink">About</a>
            <a href="${basePath}/events.html" class="nav-item">Events</a>
            <a href="${basePath}/community.html" class="nav-item">
                <span>Community</span>
            </a>
            <a href="${basePath}/arena.html" class="nav-item">
                <span>Arena</span>
            </a>
            <a href="${basePath}/dashboard.html" class="nav-item nav-item-restricted">
                <span>Dashboard</span>
                <span class="lock-container">
                    <svg class="lock-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M18 10h-1V7c0-2.76-2.24-5-5-5S7 4.24 7 7v3H6c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-8c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3-10H9V7c0-1.66 1.34-3 3-3s3 1.34 3 3v3z"/></svg>
                    <span class="lock-tooltip">Account Required</span>
                </span>
            </a>
            <a href="${basePath}/contact.html" class="nav-item hidden">Contact</a>
        </nav>

        <!-- Auth Modal -->
        <div class="auth-overlay" id="authOverlay">
            <div class="auth-modal">
                <button class="auth-close" id="authClose">&times;</button>
                <h2 class="auth-title" id="authTitle">Create Account</h2>
                
                <div class="social-login-buttons">
                    <a href="#" id="btn-login-google" class="social-btn google-btn">
                        ${googleIcon} Login with Google
                    </a>
                    <a href="#" id="btn-login-github" class="social-btn github-btn">
                        ${githubIcon} Login with GitHub
                    </a>
                </div>

                <div class="social-login-separator">
                    <span>or continue with email</span>
                </div>

                <div class="auth-error" id="authError"></div>
                <form id="authForm">
                    <input type="email" class="auth-input" id="authEmail" placeholder="Email" required>
                    <input type="password" class="auth-input" id="authPassword" placeholder="Password" required>
                    <div id="authForgotContainer" class="hidden">
                        <span class="auth-forgot-link" id="authForgotLink">Forgot Password?</span>
                    </div>
                    <button type="submit" class="auth-submit" id="authSubmit">Create Account</button>
                </form>
                <span class="auth-toggle" id="authToggle">Already have an account? Login</span>
            </div>
        </div>

        <!-- Profile Modal (for logged-in users) -->
        <div class="profile-overlay" id="profileOverlay">
            <div class="profile-modal" style="max-width: 850px; width: 95%;">
                <button class="profile-close" id="profileClose">&times;</button>
                <div style="display: flex; gap: 30px; align-items: stretch;">
                    <div style="flex: 1;">
                        <h2 class="profile-title">Your Profile</h2>
                        <form id="profileForm">
                            <div class="profile-form-group">
                                <label for="profileDisplayName" class="profile-label">Display Name:</label>
                                <input type="text" class="profile-input" id="profileDisplayName" placeholder="Display Name">
                            </div>

                            <div class="profile-form-group">
                                <label for="profileFullName" class="profile-label">Full Name:</label>
                                <input type="text" class="profile-input" id="profileFullName" placeholder="Your Full Name">
                            </div>
                            <div class="profile-form-group">
                                <label for="profileSummary" class="profile-label">Summary:</label>
                                <textarea class="profile-textarea" id="profileSummary" placeholder="A brief summary about yourself"></textarea>
                            </div>
                            <div class="profile-form-group">
                                <label for="profileRole" class="profile-label">Role:</label>
                                <select class="profile-input" id="profileRole">
                                    <option value="student" selected>Student</option>
                                    <option value="educator">Educator</option>
                                    <option value="industry">Industry</option>
                                </select>
                            </div>
                            <div class="profile-form-group">
                                <label for="profileLocation" class="profile-label">Location:</label>
                                <input type="text" class="profile-input" id="profileLocation" placeholder="City, Country">
                                <input type="hidden" id="profileLatitude">
                                <input type="hidden" id="profileLongitude">
                            </div>
                            <div class="profile-form-group">
                                <label for="profileInstitution" class="profile-label">Institution (comma-separated):</label>
                                <input type="text" class="profile-input" id="profileInstitution" placeholder="University, Company">
                            </div>
                            <div class="profile-form-group">
                                <label for="profileWebsites" class="profile-label">Websites (comma-separated URLs):</label>
                                <input type="text" class="profile-input" id="profileWebsites" placeholder="https://site1.com, https://site2.com">
                            </div>

                            <div class="profile-form-group" style="display: flex; align-items: center; gap: 10px;">
                                <input type="checkbox" id="profileMailingList" checked style="width: auto;">
                                <label for="profileMailingList" class="profile-label" style="margin: 0; font-weight: normal;">Subscribe to our mailing list</label>
                            </div>


                            <button type="submit" class="profile-submit" id="profileSubmit">Update Profile</button>
                            <button type="button" class="profile-logout-btn" id="profileLogoutBtn">Logout</button>
                        </form>
                    </div>
                    
                    <!-- Flame Side -->
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 180px; border-left: 1px solid #f0f0f0; background: rgba(252, 252, 252, 0.5);">
                         <canvas id="profileCandleCanvas" width="16" height="24" style="width: 150px; height: auto; image-rendering: pixelated; filter: drop-shadow(4px 4px 0px rgba(0,0,0,0.05));"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Message Modal (Generic) -->
        <div class="message-overlay" id="messageOverlay">
            <div class="message-modal">
                <h3 class="message-title" id="messageTitle">Notification</h3>
                <p class="message-body" id="messageBody">Message goes here.</p>
                <button class="message-btn" id="messageBtn">OK</button>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('afterbegin', layoutHTML);
    updateNavState();
    
    // Attach close handler for message modal
    const msgBtn = document.getElementById('messageBtn');
    if (msgBtn) {
        msgBtn.addEventListener('click', () => {
            document.getElementById('messageOverlay').classList.remove('active');
        });
    }
}

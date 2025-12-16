import { getBasePath } from './config.js';
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

    if (authBtnElement) {
        authBtnElement.className = btnClass;
        authBtnElement.querySelector('.login-icon').innerHTML = currentIcon;
        authBtnElement.querySelector('.btn-text').textContent = btnText;
    }

    // Show/Hide Extra Icons
    if (navDashboard) navDashboard.style.display = isLoggedIn ? 'flex' : 'none';
    if (navCommunity) navCommunity.style.display = isLoggedIn ? 'flex' : 'none';
}

export function renderLayout() {
    const basePath = getBasePath();
    const { isLoggedIn, email: userEmail } = getSession();

    const logoutIcon = `<path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>`;
    const userIcon = `<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>`;

    const footprintsIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-footprints-icon lucide-footprints"><path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 8.68V16a2 2 0 1 1-4 0Z"/><path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z"/><path d="M16 17h4"/><path d="M4 13h4"/></svg>`;
    const globeIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-earth-icon lucide-earth"><path d="M21.54 15H17a2 2 0 0 0-2 2v4.54"/><path d="M7 3.34V5a3 3 0 0 0 3 3a2 2 0 0 1 2 2c0 1.1.9 2 2 2a2 2 0 0 0 2-2c0-1.1.9-2 2-2h3.17"/><path d="M11 21.95V18a2 2 0 0 0-2-2a2 2 0 0 1-2-2v-1a2 2 0 0 0-2-2H2.05"/><circle cx="12" cy="12" r="10"/></svg>`;

    const currentIcon = isLoggedIn ? logoutIcon : userIcon;
    const btnText = isLoggedIn ? `Logout of ${userEmail}` : 'Create Account';
    const btnClass = isLoggedIn ? 'login-btn logged-in' : 'login-btn';
    const displayExtras = isLoggedIn ? 'flex' : 'none';

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
            <div class="profile-modal">
                <button class="profile-close" id="profileClose">&times;</button>
                <h2 class="profile-title">Your Profile</h2>
                <form id="profileForm">
                    <div class="profile-form-group">
                        <label for="profileDisplayName" class="profile-label">Display Name:</label>
                        <input type="text" class="profile-input" id="profileDisplayName" placeholder="Display Name">
                    </div>

                    <!-- Avatar Section -->
                    <div class="profile-form-group">
                        <label class="profile-label">Avatar:</label>
                        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                            <img id="avatarPreview" src="" alt="Preview" style="width: 60px; height: 60px; border-radius: 50%; object-fit: cover; background: #eee; border: 1px solid #ddd;">
                            <div style="display: flex; flex-direction: column; gap: 5px;">
                                <button type="button" id="btnUpload" style="font-size: 0.8rem; padding: 4px 8px; cursor: pointer;">Upload Image</button>
                                <button type="button" id="btnCamera" style="font-size: 0.8rem; padding: 4px 8px; cursor: pointer;">Use Camera</button>
                            </div>
                            <input type="file" id="fileInput" accept="image/*" style="display: none;">
                        </div>

                        <!-- Camera UI (Hidden) -->
                        <div id="cameraContainer" style="display: none; margin-bottom: 10px; text-align: center;">
                            <video id="cameraVideo" autoplay playsinline style="width: 100%; max-width: 300px; background: #000; border-radius: 8px;"></video>
                            <br>
                            <button type="button" id="btnSnap" style="margin-top: 5px; background: #ff6600; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">Take Photo</button>
                            <button type="button" id="btnStopCamera" style="margin-top: 5px; background: #666; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;">Cancel</button>
                            <canvas id="cameraCanvas" style="display: none;"></canvas>
                        </div>

                        <input type="text" class="profile-input" id="profileAvatarUrl" placeholder="https://example.com/avatar.jpg or Data URL">
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


                    <button type="submit" class="profile-submit" id="profileSubmit">Update Profile</button>
                    <button type="button" class="profile-logout-btn" id="profileLogoutBtn">Logout</button>
                </form>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('afterbegin', layoutHTML);
    updateNavState();
}

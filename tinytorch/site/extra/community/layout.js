(function() {
    // Configuration
    const SUPABASE_URL = "https://zrvmjrxhokwwmjacyhpq.supabase.co/functions/v1";
    const NETLIFY_URL = "https://tinytorch.netlify.app"; 

    // URL Base Path Logic for Community Site Hosting
    const isCommunitySite = window.location.hostname === 'tinytorch.ai' || (window.location.hostname === 'localhost' && window.location.port === '8000');
    const basePath = isCommunitySite ? '/community' : '';

    function forceLogin() {
        console.warn("Session expired or invalid. Redirecting to login...");
        localStorage.removeItem("tinytorch_token");
        localStorage.removeItem("tinytorch_refresh_token");
        localStorage.removeItem("tinytorch_user");
        window.location.href = basePath + '/index.html?action=login';
    }

    // State Management
    function getSession() {
        const token = localStorage.getItem("tinytorch_token");
        const userStr = localStorage.getItem("tinytorch_user");
        let user = null;
        try { user = JSON.parse(userStr); } catch(e) {}
        
        // Allow window.USER_EMAIL to override or serve as fallback if no local storage (legacy support)
        const email = user ? user.email : (window.USER_EMAIL || null);
        return { token, email, isLoggedIn: !!token };
    }

    // 1. Inject CSS
    const style = document.createElement('style');
    style.textContent = `
        /* --- Hamburger Menu --- */
        .menu-btn {
            position: fixed;
            top: 30px;
            left: 30px;
            z-index: 100;
            width: 35px;
            height: 25px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background: transparent;
            border: none;
            padding: 0;
        }

        .menu-btn span {
            display: block;
            width: 100%;
            height: 4px;
            background-color: #333;
            border-radius: 2px;
            transition: all 0.3s ease-in-out;
        }

        .menu-btn.active span:nth-child(1) {
            transform: translateY(10.5px) rotate(45deg);
        }
        .menu-btn.active span:nth-child(2) {
            opacity: 0;
        }
        .menu-btn.active span:nth-child(3) {
            transform: translateY(-10.5px) rotate(-45deg);
        }

        /* --- Top Right Actions Container --- */
        .top-right-actions {
            position: fixed;
            top: 30px;
            right: 30px;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* --- Login/Account Button --- */
        .login-btn {
            /* Position removed, handled by container */
            font-family: 'Verdana', sans-serif;
            font-size: 0.95rem;
            font-weight: bold;
            color: rgba(30, 30, 30, 0.8); 
            background: transparent;
            border: 2px solid rgba(30, 30, 30, 0.8); 
            text-decoration: none;
            transition: all 0.2s ease;
            cursor: pointer;
            padding: 10px 20px;
            border-radius: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .login-btn:hover {
            background: rgba(0, 0, 0, 0.05);
            color: #000;
            border-color: #000;
        }

        .login-icon {
            width: 18px;
            height: 18px;
            fill: currentColor;
        }

        /* --- Nav Icon Buttons --- */
        .nav-icon-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            color: rgba(30, 30, 30, 0.8);
            transition: all 0.2s ease;
            background: transparent;
            border: 1px solid transparent; /* Keep sizing consistent */
        }

        .nav-icon-btn:hover {
            background: rgba(0, 0, 0, 0.05);
            color: #ff6600;
        }

        /* Slide-out Sidebar (Desktop Default) */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            width: 300px;
            height: 100%;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            z-index: 90;
            transform: translateX(-100%);
            transition: transform 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
            box-shadow: 2px 0 15px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            padding-top: 100px;
            padding-left: 40px;
        }

        .sidebar.active {
            transform: translateX(0);
        }

        .nav-item {
            font-family: 'Verdana', sans-serif;
            font-size: 1.5rem;
            color: #333;
            text-decoration: none;
            margin-bottom: 20px;
            font-weight: bold;
            transition: color 0.2s;
        }

        .nav-item:hover {
            color: #ff6600;
        }

                                .nav-item-restricted {

                                    display: flex;

                                    align-items: center;

                                    position: relative; /* Add back relative positioning for tooltip */

                                    /* Existing padding from .nav-item will apply */

                                }

                

                                .lock-container {

                                    display: flex;

                                    align-items: center;

                                    /* Removed order: -1; */

                                    margin-left: 8px; /* Slightly more space between text and lock */

                                    position: relative; /* Make lock-container relative for tooltip positioning */

                                }

                

                                .lock-icon {

                                    width: 18px;

                                    height: 18px;

                                    fill: #999; /* Greyed out lock */

                                    transition: fill 0.2s;

                                }

                

                                .lock-tooltip {

                                    position: absolute;

                                    bottom: 100%; /* Position above the lock-container */

                                    left: 50%;

                                    transform: translateX(-50%) translateY(-5px); /* Center horizontally and add slight gap */

                                    background: rgba(0, 0, 0, 0.8);

                                    color: white;

                                    font-size: 0.7rem;

                                    padding: 5px 8px;

                                    border-radius: 4px;

                                    white-space: nowrap;

                                    opacity: 0;

                                    visibility: hidden;

                                    transition: opacity 0.2s, visibility 0.2s;

                                    pointer-events: none;

                                    z-index: 10;

                                }

                

                                /* Hover on parent nav-item-restricted shows tooltip */

                                .nav-item-restricted:hover .lock-tooltip {

                                    opacity: 1;

                                    visibility: visible;

                                }

                

                                .nav-item-restricted:hover .lock-icon {

                                    fill: #ff6600; /* Orange lock on hover */

                                }

                

                /* --- Modal Styles --- */
        .auth-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
            z-index: 200;
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s;
        }
        
        .auth-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .auth-modal {
            background: white;
            padding: 40px;
            border-radius: 20px;
            width: 90%;
            max-width: 400px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: relative;
            transform: translateY(20px);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .auth-overlay.active .auth-modal {
            transform: translateY(0);
        }

        .auth-close {
            position: absolute;
            top: 20px;
            right: 20px;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #888;
        }

        .auth-title {
            font-family: 'Verdana', sans-serif;
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }

        .auth-input {
            width: 100%;
            padding: 12px 15px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 10px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
            box-sizing: border-box;
        }

        .auth-input:focus {
            border-color: #ff6600;
        }

        .auth-submit {
            width: 100%;
            padding: 12px;
            background: #333;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
        }

        .auth-submit:hover {
            background: #ff6600;
        }

        .auth-toggle {
            display: block;
            text-align: center;
            margin-top: 15px;
            color: #666;
            text-decoration: underline;
            cursor: pointer;
            font-size: 0.9rem;
        }
        
        .auth-error {
            color: #d32f2f;
            font-size: 0.9rem;
            margin-bottom: 15px;
            text-align: center;
            display: none;
        }

        .auth-forgot-link {
            display: block;
            text-align: right;
            margin-top: -10px;
            margin-bottom: 15px;
            font-size: 0.85rem;
            color: #666;
            text-decoration: none;
            cursor: pointer;
        }
        .auth-forgot-link:hover {
            color: #ff6600;
            text-decoration: underline;
        }

        .hidden {
            display: none !important;
        }

        /* --- Profile Modal Styles --- */
        .profile-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
            z-index: 200;
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s;
        }

        .profile-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .profile-modal {
            background: white;
            padding: 30px;
            border-radius: 20px;
            width: 90%;
            max-width: 500px; /* Slightly wider for profile fields */
            max-height: 80vh; /* Cap modal height strictly */
            overflow-y: auto; /* Enable scrolling for overflow content */
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: relative;
            transform: translateY(20px);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .profile-overlay.active .profile-modal {
            transform: translateY(0);
        }

        .profile-close {
            position: absolute;
            top: 20px;
            right: 20px;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #888;
        }

        .profile-title {
            font-family: 'Verdana', sans-serif;
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }

        .profile-form-group {
            margin-bottom: 15px;
        }

        .profile-label {
            display: block;
            margin-bottom: 5px;
            font-family: 'Verdana', sans-serif;
            font-size: 0.9rem;
            color: #555;
        }

        .profile-input, .profile-textarea {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 10px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
            box-sizing: border-box;
        }

        .profile-textarea {
            min-height: 80px;
            resize: vertical;
        }

        .profile-input:focus, .profile-textarea:focus {
            border-color: #ff6600;
        }

        .profile-submit {
            width: 100%;
            padding: 12px;
            background: #ff6600; /* Orange for update action */
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 20px;
        }

        .profile-submit:hover {
            background: #e65c00;
        }

        .profile-logout-btn {
            width: 100%;
            padding: 12px;
            background: #d32f2f; /* Red for logout */
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 10px;
        }

        .profile-logout-btn:hover {
            background: #b71c1c;
        }

        /* --- Mobile Optimizations --- */
        @media (max-width: 768px) {
            /* Sidebar becomes a bottom sheet */
            .sidebar {
                top: auto;
                bottom: 0;
                left: 0;
                width: 100%;
                height: auto;
                transform: translateY(100%);
                border-radius: 24px 24px 0 0;
                padding: 40px 20px 80px 20px; 
                box-shadow: 0 -5px 20px rgba(0,0,0,0.15);
                align-items: center; 
                padding-left: 0;     
            }

            .sidebar.active {
                transform: translateY(0);
            }
            
            .login-btn.logged-in {
                padding: 0;
                width: 45px;
                height: 45px;
                justify-content: center;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(0,0,0,0.1);
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            .login-btn.logged-in .btn-text {
                display: none;
            }

            .login-btn.logged-in .login-icon {
                width: 24px;
                height: 24px;
                margin: 0;
            }
            
            .menu-btn {
                top: auto;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 100px; 
                height: 25px; 
                background: rgba(255, 255, 255, 0.95);
                border-radius: 16px 16px 0 0;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
                padding: 0; 
                justify-content: center; 
                align-items: center; 
                border: 1px solid rgba(0,0,0,0.05);
                border-bottom: none;
                backdrop-filter: blur(5px);
            }

            .menu-btn span {
                width: 40px; 
                height: 4px; 
                background: #ccc;
                border-radius: 2px;
                margin: 0; 
            }

            .menu-btn span:nth-child(2),
            .menu-btn span:nth-child(3) {
                display: none;
            }

            .menu-btn.active span:nth-child(1) {
                transform: none;
                background: #ff6600; 
            }
            
            .top-right-actions {
                top: 20px;
                right: 20px;
            }
        }

    `;
    document.head.appendChild(style);

    // 2. Create and Inject HTML Elements
    
    const userIcon = `<path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>`;
    const logoutIcon = `<path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z"/>`;
    
    const footprintsIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-footprints-icon lucide-footprints"><path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 8.68V16a2 2 0 1 1-4 0Z"/><path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z"/><path d="M16 17h4"/><path d="M4 13h4"/></svg>`;
    const globeIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-earth-icon lucide-earth"><path d="M21.54 15H17a2 2 0 0 0-2 2v4.54"/><path d="M7 3.34V5a3 3 0 0 0 3 3a2 2 0 0 1 2 2c0 1.1.9 2 2 2a2 2 0 0 0 2-2c0-1.1.9-2 2-2h3.17"/><path d="M11 21.95V18a2 2 0 0 0-2-2a2 2 0 0 1-2-2v-1a2 2 0 0 0-2-2H2.05"/><circle cx="12" cy="12" r="10"/></svg>`;

    // Initialize button state based on current session
    function updateNavState() {
        const { isLoggedIn, email: userEmail } = getSession();
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

    // Define initial state variables for layoutHTML
    const { isLoggedIn, email: userEmail } = getSession();
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

            <a href="${basePath}/community.html" class="nav-icon-btn" id="navCommunityBtn" style="display: ${displayExtras};" title="Community">
                ${globeIcon}
            </a>
        </div>

        <nav class="sidebar" id="sidebar">
            <a href="${basePath}/index.html" class="nav-item">Home</a>
            <a href="${basePath}/about.html" class="nav-item" id="aboutLink">About</a>
            <a href="${basePath}/events.html" class="nav-item">Events</a>
            <a href="${basePath}/community.html" class="nav-item nav-item-restricted">
                <span>Community</span>
                <span class="lock-container">
                    <svg class="lock-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M18 10h-1V7c0-2.76-2.24-5-5-5S7 4.24 7 7v3H6c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-8c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3-10H9V7c0-1.66 1.34-3 3-3s3 1.34 3 3v3z"/></svg>
                    <span class="lock-tooltip">Account Required</span>
                </span>
            </a>
            <a href="${basePath}/dashboard.html" class="nav-item nav-item-restricted">
                <span>Dashboard</span>
                <span class="lock-container">
                    <svg class="lock-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M18 10h-1V7c0-2.76-2.24-5-5-5S7 4.24 7 7v3H6c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-8c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3-10H9V7c0-1.66 1.34-3 3-3s3 1.34 3 3v3z"/></svg>
                    <span class="lock-tooltip">Account Required</span>
                </span>
            </a>
            <a href="${basePath}/contact.html" class="nav-item">Contact</a>
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
                    </div>
                    <div class="profile-form-group">
                        <label for="profileInstitution" class="profile-label">Institution (comma-separated):</label>
                        <input type="text" class="profile-input" id="profileInstitution" placeholder="University, Company">
                    </div>
                    <div class="profile-form-group">
                        <label for="profileWebsites" class="profile-label">Websites (comma-separated URLs):</label>
                        <input type="text" class="profile-input" id="profileWebsites" placeholder="https://site1.com, https://site2.com">
                    </div>
                    <div class="profile-form-group">
                        <label for="profileContactJson" class="profile-label">Contact Info (JSON):</label>
                        <textarea class="profile-textarea" id="profileContactJson" placeholder='{"phone": "+123456789", "twitter": "@handle"}'></textarea>
                    </div>
                    <div class="profile-form-group">
                        <label for="profilePreferences" class="profile-label">Preferences (JSON):</label>
                        <textarea class="profile-textarea" id="profilePreferences" placeholder='{"theme": "dark", "notifications": true}'></textarea>
                    </div>
                    <div class="profile-form-group">
                        <input type="checkbox" id="profileIsPublic">
                        <label for="profileIsPublic"> Make profile public</label>
                    </div>

                    <button type="submit" class="profile-submit" id="profileSubmit">Update Profile</button>
                    <button type="button" class="profile-logout-btn" id="profileLogoutBtn">Logout</button>
                </form>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('afterbegin', layoutHTML);
    updateNavState(); // Initialize the button state


    // 3. Add Event Listeners
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.getElementById('sidebar');
    const authBtn = document.getElementById('authBtn');
    const authOverlay = document.getElementById('authOverlay');
    const authClose = document.getElementById('authClose');
    const authForm = document.getElementById('authForm');
    const authTitle = document.getElementById('authTitle');
    const authSubmit = document.getElementById('authSubmit');
    const authToggle = document.getElementById('authToggle');
    const authError = document.getElementById('authError');
    const emailInput = document.getElementById('authEmail');
    const passwordInput = document.getElementById('authPassword');
    const forgotLink = document.getElementById('authForgotLink');
    const forgotContainer = document.getElementById('authForgotContainer');

    // Profile Modal Elements
    const profileOverlay = document.getElementById('profileOverlay');
    const profileClose = document.getElementById('profileClose');
    const profileForm = document.getElementById('profileForm');
    const profileDisplayNameInput = document.getElementById('profileDisplayName');
    const profileAvatarUrlInput = document.getElementById('profileAvatarUrl');
    const profileIsPublicCheckbox = document.getElementById('profileIsPublic');
    const profileFullNameInput = document.getElementById('profileFullName');
    const profileSummaryTextarea = document.getElementById('profileSummary');
    const profileContactJsonTextarea = document.getElementById('profileContactJson');
    const profileLocationInput = document.getElementById('profileLocation');
    const profileWebsitesInput = document.getElementById('profileWebsites');
    const profileInstitutionInput = document.getElementById('profileInstitution');
    const profilePreferencesTextarea = document.getElementById('profilePreferences');
    const profileSubmitBtn = document.getElementById('profileSubmit');
    const profileLogoutBtn = document.getElementById('profileLogoutBtn');

    // --- Avatar Logic ---
    const avatarPreview = document.getElementById('avatarPreview');
    const btnUpload = document.getElementById('btnUpload');
    const btnCamera = document.getElementById('btnCamera');
    const fileInput = document.getElementById('fileInput');
    const cameraContainer = document.getElementById('cameraContainer');
    const cameraVideo = document.getElementById('cameraVideo');
    const btnSnap = document.getElementById('btnSnap');
    const btnStopCamera = document.getElementById('btnStopCamera');
    const cameraCanvas = document.getElementById('cameraCanvas');
    let mediaStream = null;

    // Helper to resize images
    function resizeImage(file, maxWidth, maxHeight, quality) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    let width = img.width;
                    let height = img.height;

                    if (width > height) {
                        if (width > maxWidth) {
                            height = Math.round((height * maxWidth) / width);
                            width = maxWidth;
                        }
                    } else {
                        if (height > maxHeight) {
                            width = Math.round((width * maxHeight) / height);
                            height = maxHeight;
                        }
                    }

                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    resolve(canvas.toDataURL('image/jpeg', quality));
                };
                img.onerror = (err) => reject(err);
                img.src = event.target.result;
            };
            reader.onerror = (err) => reject(err);
            reader.readAsDataURL(file);
        });
    }

    // 1. Preview on Input Change
    if (profileAvatarUrlInput) {
        profileAvatarUrlInput.addEventListener('input', () => {
            avatarPreview.src = profileAvatarUrlInput.value || ''; // Fallback to placeholder or empty
        });
    }

    // 2. File Upload
    if (btnUpload && fileInput) {
        btnUpload.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    // Resize to thumbnail size (e.g., 300x300)
                    const result = await resizeImage(file, 300, 300, 0.7);
                    profileAvatarUrlInput.value = result;
                    avatarPreview.src = result;
                } catch (err) {
                    console.error("Error resizing image:", err);
                    alert("Failed to process image. Please try another file.");
                }
            }
        });
    }

    // 3. Camera Logic
    async function startCamera() {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraVideo.srcObject = mediaStream;
            cameraContainer.style.display = 'block';
        } catch (err) {
            console.error("Camera access denied:", err);
            alert("Could not access camera. Please check permissions.");
        }
    }

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        cameraContainer.style.display = 'none';
    }

    if (btnCamera) {
        btnCamera.addEventListener('click', startCamera);
    }
    
    if (btnStopCamera) {
        btnStopCamera.addEventListener('click', stopCamera);
    }

    if (btnSnap) {
        btnSnap.addEventListener('click', () => {
            if (!mediaStream) return;
            
            // Set canvas dims to video dims but constrained
            let w = cameraVideo.videoWidth;
            let h = cameraVideo.videoHeight;
            
            // Constrain to 300x300 max while preserving aspect ratio
            const MAX_DIM = 300;
            if (w > h) {
                if (w > MAX_DIM) {
                    h = Math.round((h * MAX_DIM) / w);
                    w = MAX_DIM;
                }
            } else {
                if (h > MAX_DIM) {
                    w = Math.round((w * MAX_DIM) / h);
                    h = MAX_DIM;
                }
            }

            cameraCanvas.width = w;
            cameraCanvas.height = h;
            
            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraVideo, 0, 0, w, h);
            
            // Convert to Data URL (JPEG for smaller size)
            const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.7);
            
            profileAvatarUrlInput.value = dataUrl;
            avatarPreview.src = dataUrl;
            
            stopCamera();
        });
    }

    // Update populate to set initial preview
    const originalPopulate = populateProfileForm;
    populateProfileForm = function(data) {
        originalPopulate(data);
        if(avatarPreview) {
             avatarPreview.src = data.avatar || ''; // Set initial preview
        }
    };

    // Modal State
    let currentMode = 'signup'; // 'login', 'signup', 'forgot'

    function openModal(mode = 'signup') { // Added mode parameter with default
        authOverlay.classList.add('active');
        setMode(mode); // Use the provided mode
    }

    function closeModal() {
        authOverlay.classList.remove('active');
    }

    function openProfileModal() {
        profileOverlay.classList.add('active');
        fetchUserProfile();
    }

    async function fetchUserProfile() {
        let token = localStorage.getItem("tinytorch_token"); 
        if (!token) {
            console.error("No token found for fetching profile.");
            forceLogin(); 
            return;
        }

        let profileData = null;
        let retryCount = 0;
        const MAX_RETRIES = 1; 

        do {
            try {
                const response = await fetch(`${SUPABASE_URL}/get-profile-details`, { 
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                });

                // Check for 401 OR 400 with specific "Invalid Token" error
                let needsRefresh = response.status === 401;
                let errorData = null;

                if (response.status === 400) {
                    // Clone response to safely check body for specific error message
                    try {
                        errorData = await response.clone().json();
                        if (errorData && errorData.error && errorData.error.includes("Invalid Token")) {
                            needsRefresh = true;
                        }
                    } catch(e) { /* ignore parse error */ }
                }

                if (needsRefresh && retryCount === 0) {
                    console.log("Token expired or invalid (400/401). Attempting refresh...");
                    const refreshToken = localStorage.getItem("tinytorch_refresh_token");
                    if (!refreshToken) { forceLogin(); return; }

                    const refreshRes = await fetch(`${NETLIFY_URL}/api/auth/refresh`, { 
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ refreshToken })
                    });

                    if (!refreshRes.ok) { forceLogin(); return; }

                    const refreshData = await refreshRes.json();
                    if (refreshData.session) {
                        token = refreshData.session.access_token;
                        localStorage.setItem("tinytorch_token", token);
                        if (refreshData.session.refresh_token) {
                            localStorage.setItem("tinytorch_refresh_token", refreshData.session.refresh_token);
                        }
                        retryCount++; 
                        continue; 
                    } else {
                        forceLogin();
                        return;
                    }
                }

                if (!response.ok) {
                    // If errorData wasn't parsed yet (e.g. not a 400 or parse failed), try now
                    if (!errorData) {
                         try { errorData = await response.json(); } catch(e) {}
                    }
                    throw new Error(errorData?.error || `Failed to fetch profile data: ${response.status}`);
                }

                profileData = await response.json();
                populateProfileForm(profileData.profile); 
                return; 
            } catch (error) {
                console.error("Error fetching user profile:", error);
                // Don't alert immediately on first fail if we might be refreshing, 
                // but here we are inside the catch, meaning a hard error occurred.
                if (retryCount >= MAX_RETRIES || !error.message.includes("Invalid Token")) {
                    alert("Failed to load profile data. Please try again.");
                    closeProfileModal();
                }
                return; 
            }
        } while (retryCount < MAX_RETRIES);

        if (!profileData) {
            alert("Failed to load profile data after multiple attempts. Please try again.");
            closeProfileModal();
        }
    }

    function populateProfileForm(data) {
        // Map API response keys to Form Input IDs
        // Response: avatar, bio, socials, institution (array), websites (array), display_name, preferences, is_public
        
        profileDisplayNameInput.value = data.display_name || '';
        profileAvatarUrlInput.value = data.avatar || data.avatar_url || ''; // Support both
        profileIsPublicCheckbox.checked = data.is_public || false;
        profileFullNameInput.value = data.full_name || '';
        profileSummaryTextarea.value = data.bio || data.summary || ''; // Support both
        profileLocationInput.value = data.location || '';
        
        // Handle Arrays
        profileInstitutionInput.value = Array.isArray(data.institution) ? data.institution.join(', ') : (data.institution || '');
        
        const sites = data.website || data.websites; // Prioritize 'website' based on DB response
        profileWebsitesInput.value = Array.isArray(sites) ? sites.join(', ') : (sites || '');
        
        // Handle JSON objects
        try {
            profileContactJsonTextarea.value = data.socials ? JSON.stringify(data.socials, null, 2) : ''; // API returns 'socials'
        } catch (e) {
            console.error("Error parsing socials:", e);
            profileContactJsonTextarea.value = '';
        }

        try {
            profilePreferencesTextarea.value = data.preferences ? JSON.stringify(data.preferences, null, 2) : '{"theme": "standard"}';
        } catch (e) {
            console.error("Error parsing preferences:", e);
            profilePreferencesTextarea.value = '{"theme": "standard"}';
        }
    }

    async function handleProfileUpdate(e) {
        e.preventDefault();
        let token = localStorage.getItem("tinytorch_token"); 
        if (!token) {
            console.error("No token found for updating profile.");
            forceLogin();
            return;
        }

        const updatedProfile = {
            display_name: profileDisplayNameInput.value,
            avatar: profileAvatarUrlInput.value, // Changed from avatar_url to match GET response
            is_public: profileIsPublicCheckbox.checked,
            full_name: profileFullNameInput.value,
            summary: profileSummaryTextarea.value,
            location: profileLocationInput.value,
            // Split string inputs into arrays
            institution: profileInstitutionInput.value.split(',').map(s => s.trim()).filter(s => s),
            website: profileWebsitesInput.value.split(',').map(s => s.trim()).filter(s => s), // Changed from websites to website
        };
        
        console.log("Sending Profile Update Payload:", updatedProfile); // Debug payload

        try {
            updatedProfile.contact_json = profileContactJsonTextarea.value ? JSON.parse(profileContactJsonTextarea.value) : null;
        } catch (e) {
            alert("Invalid Contact Info JSON. Please correct it.");
            return;
        }
        try {
            updatedProfile.preferences = profilePreferencesTextarea.value ? JSON.parse(profilePreferencesTextarea.value) : '{"theme": "standard"}';
        } catch (e) {
            alert("Invalid Preferences JSON. Please correct it.");
            return;
        }

        let retryCount = 0;
        const MAX_RETRIES = 1; 

        do {
            try {
                const response = await fetch(`${SUPABASE_URL}/update-profile`, { 
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(updatedProfile)
                });

                let needsRefresh = response.status === 401;
                let errorData = null;

                if (response.status === 400) {
                    try {
                        errorData = await response.clone().json();
                        if (errorData && errorData.error && errorData.error.includes("Invalid Token")) {
                            needsRefresh = true;
                        }
                    } catch(e) {}
                }

                if (needsRefresh && retryCount === 0) {
                    console.log("Token expired or invalid during update (400/401). Attempting refresh...");
                    const refreshToken = localStorage.getItem("tinytorch_refresh_token");
                    if (!refreshToken) { forceLogin(); return; }

                    const refreshRes = await fetch(`${NETLIFY_URL}/api/auth/refresh`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ refreshToken })
                    });

                    if (!refreshRes.ok) { forceLogin(); return; }

                    const refreshData = await refreshRes.json();
                    if (refreshData.session) {
                        token = refreshData.session.access_token;
                        localStorage.setItem("tinytorch_token", token);
                        if (refreshData.session.refresh_token) {
                            localStorage.setItem("tinytorch_refresh_token", refreshData.session.refresh_token);
                        }
                        retryCount++;
                        continue;
                    } else {
                        forceLogin();
                        return;
                    }
                }

                if (!response.ok) {
                    if (!errorData) {
                         try { errorData = await response.json(); } catch(e) {}
                    }
                    throw new Error(errorData?.error || 'Failed to update profile');
                }

                alert('Profile updated successfully!');
                closeProfileModal();
                return; 
            } catch (error) {
                console.error("Error updating user profile:", error);
                alert("Failed to update profile: " + error.message);
                return; 
            }
        } while (retryCount < MAX_RETRIES);
    }

    function closeProfileModal() {
        profileOverlay.classList.remove('active');
    }

    function resetForm() {
        authForm.reset();
        authError.style.display = 'none';
        authError.textContent = '';
        if (currentMode === 'forgot') {
            emailInput.value = ''; // Clear email for forgot password mode
        }
    }

    function setMode(mode) {
        const previousEmail = emailInput.value; // Store email before reset
        currentMode = mode;
        resetForm();
        
        if (mode === 'login' || mode === 'signup') { // Restore email for login/signup modes
            emailInput.value = previousEmail;
        }

        if (mode === 'login') {
            authTitle.textContent = 'Login';
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

    function handleToggle() {
        if (currentMode === 'login') {
            setMode('signup');
        } else if (currentMode === 'signup') {
            setMode('login');
        } else if (currentMode === 'forgot') {
            setMode('login');
        }
    }

    // Auth Logic
    async function handleAuth(e) {
        e.preventDefault();
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
                body = { email, password };
            }
            
            const url = `${NETLIFY_URL}${endpoint}`;

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });

            const data = await response.json();

            // Check for success based on mode
            if (currentMode === 'forgot') {
                 // Forgot password always returns 200 OK with message, or 400 error
                 if (response.ok) {
                     alert(data.message || 'If an account exists, a reset link has been sent.');
                     setMode('login'); // Go back to login
                 } else {
                     throw new Error(data.error || 'Failed to send reset link');
                 }
            } else {
                if (!response.ok) {
                    throw new Error(data.error || (currentMode === 'login' ? 'Login failed' : 'Signup failed'));
                }

                if (currentMode === 'login') {
                    // Login Success
                    if (data.access_token) {
                        localStorage.setItem("tinytorch_token", data.access_token);
                        if (data.refresh_token) localStorage.setItem("tinytorch_refresh_token", data.refresh_token);
                        localStorage.setItem("tinytorch_user", JSON.stringify(data.user));
                        updateNavState(); // Update button state
                        
                        const params = new URLSearchParams(window.location.search);
                        const redirectAction = params.get('action') === 'profile' ? '?action=profile' : '';
                        const next = params.get('next');
                        if (next) {
                            window.location.href = basePath + '/' + next + redirectAction;
                        } else {
                            window.location.href = basePath + '/dashboard.html' + redirectAction;
                        }
                    }
                } else {
                    // Signup Success
                    alert('Account created successfully! Please check your email to confirm before logging in.');
                    setMode('login'); // Switch to login
                    updateNavState(); // Update button state
                }
            }

        } catch (error) {
            console.error("Auth error:", error);
            authError.textContent = error.message;
            authError.style.display = 'block';
        } finally {
            authSubmit.disabled = false;
            // Restore button text based on mode
            if (currentMode === 'login') authSubmit.textContent = 'Login';
            else if (currentMode === 'signup') authSubmit.textContent = 'Create Account';
            else authSubmit.textContent = 'Send Reset Link';
        }
    }

    function handleLogout() {
        if (confirm('Are you sure you want to logout?')) {
            localStorage.removeItem("tinytorch_token");
            localStorage.removeItem("tinytorch_user");
            updateNavState(); // Update button state
            window.location.href = basePath + '/';
        }
    }

    // Wire up events
    if (menuBtn && sidebar) {
        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('active') && 
                !sidebar.contains(e.target) && 
                !menuBtn.contains(e.target)) {
                menuBtn.classList.remove('active');
                sidebar.classList.remove('active');
            }
        });
        
        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menuBtn.classList.toggle('active');
            sidebar.classList.toggle('active');
        });
    }

    if (authBtn) {
        authBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const { isLoggedIn } = getSession(); // Get current state
            if (isLoggedIn) {
                openProfileModal(); // Open profile modal for logged-in users
            } else {
                openModal(); // Open auth modal for logged-out users
            }
        });
    }

    if (authClose) authClose.addEventListener('click', closeModal);
    if (authOverlay) {
        authOverlay.addEventListener('click', (e) => {
            if (e.target === authOverlay) closeModal();
        });
    }
    if (authToggle) authToggle.addEventListener('click', handleToggle);
    if (forgotLink) forgotLink.addEventListener('click', () => setMode('forgot'));
    if (authForm) authForm.addEventListener('submit', handleAuth);

    // Profile Modal Events
    if (profileClose) profileClose.addEventListener('click', closeProfileModal);
    if (profileOverlay) {
        profileOverlay.addEventListener('click', (e) => {
            if (e.target === profileOverlay) closeProfileModal();
        });
    }
    if (profileLogoutBtn) profileLogoutBtn.addEventListener('click', handleLogout);
    if (profileForm) profileForm.addEventListener('submit', handleProfileUpdate);

    // Check for redirect action
    const params = new URLSearchParams(window.location.search);
    const action = params.get('action');
    
    if (action === 'login') {
        // Clear any stale tokens just in case
        localStorage.removeItem("tinytorch_token");
        localStorage.removeItem("tinytorch_refresh_token");
        localStorage.removeItem("tinytorch_user");
        updateNavState();
        openModal('login'); // Explicitly open in login mode
    } else if (action === 'profile') {
        const { isLoggedIn } = getSession();
        if (isLoggedIn) {
            openProfileModal();
        } else {
            openModal('login');
        }
    } else if (action === 'join') {
        const { isLoggedIn } = getSession();
        if (!isLoggedIn) {
            openModal('signup');
        }
    }

})();

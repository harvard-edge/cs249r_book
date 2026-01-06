export function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
        /* --- Hamburger Menu --- */
        .menu-btn {
            position: fixed;
            top: 30px;
            left: 30px;
            z-index: 120;
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
            z-index: 120;
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
            z-index: 110;
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

        /* --- Social Login Styles --- */
        .social-login-separator {
            display: flex;
            align-items: center;
            text-align: center;
            margin: 20px 0;
            color: #888;
            font-size: 0.85rem;
        }
        .social-login-separator::before,
        .social-login-separator::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #ddd;
        }
        .social-login-separator span {
            padding: 0 10px;
        }

        .social-login-buttons {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }

        .social-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 10px;
            border-radius: 10px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
            font-size: 0.95rem;
            border: 1px solid #ddd;
        }

        .social-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .google-btn {
            background: #fff;
            color: #333;
        }
        .github-btn {
            background: #24292e;
            color: #fff;
            border-color: #24292e;
        }
        
        .social-icon {
            width: 20px;
            height: 20px;
        }

        /* --- Spinner --- */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* --- Message Modal --- */
        .message-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
            z-index: 250; /* Higher than auth modal */
            display: flex;
            justify-content: center;
            align-items: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s;
        }
        .message-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        .message-modal {
            background: white;
            padding: 30px;
            border-radius: 20px;
            width: 90%;
            max-width: 400px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            text-align: center;
            transform: translateY(20px);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .message-overlay.active .message-modal {
            transform: translateY(0);
        }
        .message-title {
            font-family: 'Verdana', sans-serif;
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #333;
            font-weight: bold;
        }
        .message-body {
            font-family: 'Verdana', sans-serif;
            font-size: 1rem;
            color: #555;
            margin-bottom: 25px;
            line-height: 1.5;
        }
        .message-btn {
            background: #333;
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 20px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .message-btn:hover {
            background: #ff6600;
        }
    `;
    document.head.appendChild(style);
}

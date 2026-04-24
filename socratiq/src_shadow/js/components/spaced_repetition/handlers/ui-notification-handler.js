export class UINotificationHandler {
    constructor(modal) {
        this.modal = modal;
        this.shadowRoot = modal?.shadowRoot;
        this.activeNotifications = new Map();
        this.createNotificationContainer();
    }

    createNotificationContainer() {
        const root = this.shadowRoot || document.body;
        
        if (!root.querySelector('#sr-notification-container')) {
            const container = document.createElement('div');
            container.id = 'sr-notification-container';
            container.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 99999;
                display: flex;
                flex-direction: column;
                gap: 8px;
                pointer-events: none;
            `;
            root.appendChild(container);
        } else {
            console.warn("Notification container already exists");
        }
    }

    showSavingNotification(message = 'Saving...') {
        const notification = document.createElement('div');
        const id = 'notification-' + Date.now();
        notification.id = id;
        notification.style.cssText = `
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 200px;
            animation: slideIn 0.3s ease-out;
        `;
        notification.innerHTML = `
            <style>
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                .sr-spinner {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #e2e8f0;
                    border-top-color: #3b82f6;
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                }
            </style>
            <div class="sr-spinner"></div>
            <span style="color: #4b5563;">${message}</span>
        `;
        
        const container = (this.shadowRoot || document).querySelector('#sr-notification-container');
        container.appendChild(notification);
        return id;
    }

    updateNotification(id, message, type = 'success') {
        if (!id) return;
        
        // Clear existing timeout
        if (this.activeNotifications.has(id)) {
            clearTimeout(this.activeNotifications.get(id));
            this.activeNotifications.delete(id);
        }

        const notification = (this.shadowRoot || document).getElementById(id);
        if (notification) {
            const bgColor = type === 'success' ? '#dcfce7' : type === 'error' ? '#fee2e2' : '#fff';
            const textColor = type === 'success' ? '#166534' : type === 'error' ? '#991b1b' : '#4b5563';
            const icon = type === 'success' ? 
                '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>' :
                type === 'error' ? 
                '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>' :
                '';
            
            notification.style.background = bgColor;
            notification.innerHTML = `
                <div style="color: ${textColor}; padding: 4px; display: flex; align-items: center; gap: 8px;">
                    ${icon}
                    <span>${message}</span>
                </div>
            `;
            
            // Remove after delay (for success/error messages)
            const timeout = setTimeout(() => {
                this.removeLoadingState(id);
            }, 3000);
            
            this.activeNotifications.set(id, timeout);
        }
    }

    showLoadingState(message = 'Loading...') {
        const container = (this.shadowRoot || document).querySelector('#sr-notification-container');
        if (!container) {
            console.error('No notification container found');
            return null;
        }

        const id = 'loading-state-' + Date.now();
        const notification = document.createElement('div');
        notification.id = id;
        notification.className = 'sr-notification loading';
        notification.style.cssText = `
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 200px;
            animation: slideIn 0.3s ease-out;
        `;
        
        notification.innerHTML = `
            <style>
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOut {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                .sr-spinner {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #e2e8f0;
                    border-top-color: #3b82f6;
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                }
            </style>
            <div class="sr-spinner"></div>
            <span style="color: #4b5563;">${message}</span>
        `;
        
        container.appendChild(notification);

        // Set maximum timeout of 5 seconds - silently remove the loading state
        const timeout = setTimeout(() => {
            this.removeLoadingState(id);
        }, 10000);
        
        this.activeNotifications.set(id, timeout);
        return id;
    }

    removeLoadingState(id) {
        // Clear any existing timeout
        if (this.activeNotifications.has(id)) {
            clearTimeout(this.activeNotifications.get(id));
            this.activeNotifications.delete(id);
        }

        const notification = (this.shadowRoot || document).getElementById(id);
        if (notification) {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => notification.remove(), 300);
        }
    }
}

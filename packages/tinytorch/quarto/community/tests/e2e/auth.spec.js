const { test, expect } = require('@playwright/test');

test.describe('Authentication Flow', () => {
  
  test('Landing page should be accessible without login', async ({ page }) => {
    await page.goto('index.html');
    await expect(page).toHaveTitle(/AI History Landscape|The Tiny Torch/);
    // Check if login button is visible
    const authBtn = page.locator('#authBtn');
    await expect(authBtn).toBeVisible();
  });

  test('Protected page (dashboard) should redirect to index with login action', async ({ page }) => {
    await page.goto('dashboard.html');
    // Guard should redirect to index.html?action=login&next=...
    await expect(page).toHaveURL(/index\.html\?action=login/);
    
    // Auth modal should be active
    const authOverlay = page.locator('#authOverlay');
    await expect(authOverlay).toHaveClass(/active/);
  });

  test('Manual Email Login - UI Interaction', async ({ page }) => {
    await page.goto('index.html?action=login');
    
    // Switch to login mode if it defaults to signup
    const toggle = page.locator('#authToggle');
    const title = page.locator('#authTitle');
    
    // Wait for modal to be ready
    await expect(page.locator('#authOverlay')).toHaveClass(/active/);
    
    if ((await title.innerText()) === 'Create Account') {
        await toggle.click();
    }
    
    await expect(title).toHaveText('Login');

    // Fill in credentials (using dummy ones for UI test)
    await page.fill('#authEmail', 'test@example.com');
    await page.fill('#authPassword', 'password123');
    
    // Click login
    const loginBtn = page.locator('#authSubmit');
    await expect(loginBtn).toHaveText('Login');
  });

  test('Logout should clear session and redirect to index', async ({ page }) => {
    // Manually set a mock session
    await page.goto('index.html');
    await page.evaluate(() => {
      localStorage.setItem('tinytorch_token', 'mock-token');
      localStorage.setItem('tinytorch_user', JSON.stringify({ email: 'test@example.com' }));
    });
    
    await page.reload();
    
    // Open profile/logout modal
    await page.click('#authBtn');
    
    // Listen for dialog (confirm logout)
    page.on('dialog', dialog => dialog.accept());
    
    const logoutBtn = page.locator('#profileLogoutBtn');
    await expect(logoutBtn).toBeVisible();
    
    // Wait for navigation after clicking logout
    await Promise.all([
        page.waitForURL(/index\.html/),
        logoutBtn.click()
    ]);
    
    // Verify session is completely purged
    const token = await page.evaluate(() => localStorage.getItem('tinytorch_token'));
    expect(token).toBeNull();
  });
});

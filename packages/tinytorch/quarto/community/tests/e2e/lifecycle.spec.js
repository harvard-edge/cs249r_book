const { test, expect } = require('@playwright/test');
const { testUser } = require('./credentials.json');

test.describe('User Account Lifecycle', () => {

  test('Login and Navigate Profile', async ({ page }) => {
    // 1. Navigate to login
    await page.goto('index.html?action=login');
    
    // Ensure we are in login mode
    const title = page.locator('#authTitle');
    if ((await title.innerText()) === 'Create Account') {
        await page.click('#authToggle');
    }

    // 2. Perform Login
    await page.fill('#authEmail', testUser.email);
    await page.fill('#authPassword', testUser.password);
    
    // We expect a redirect after login (to dashboard or profile_setup)
    await Promise.all([
      page.waitForURL(/dashboard\.html|profile_setup\.html/),
      page.click('#authSubmit')
    ]);

    console.log('✅ Logged in successfully');

    // 3. Verify we can open the profile modal
    // Note: If redirect went to profile_setup.html, authBtn might already be active
    await page.click('#authBtn'); 
    
    const profileOverlay = page.locator('#profileOverlay');
    await expect(profileOverlay).toHaveClass(/active/);
    
    // 4. Check if the display name is loaded correctly
    const displayNameInput = page.locator('#profileDisplayName');
    await expect(displayNameInput).not.toHaveValue('');
    
    console.log('✅ Profile data loaded correctly');

    // 5. Navigate to Dashboard
    await page.goto('dashboard.html');
    await expect(page).toHaveURL(/dashboard\.html/);
    
    console.log('✅ Navigation verified');
  });

});

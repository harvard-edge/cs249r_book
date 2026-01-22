import { SUPABASE_URL, NETLIFY_URL, getBasePath } from './config.js';
import { forceLogin } from './state.js?v=2';
import { initCandle } from './candle.js';

export async function geocodeAndSetCoordinates(location) {
    const latInput = document.getElementById('profileLatitude');
    const lonInput = document.getElementById('profileLongitude');

    if (!location) {
        if(latInput) latInput.value = '';
        if(lonInput) lonInput.value = '';
        return;
    }

    try {
        const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(location)}&limit=1`);
        if (!response.ok) {
            throw new Error('Geocoding search failed');
        }
        const data = await response.json();
        if (data && data.length > 0) {
            const { lat, lon } = data[0];
            if(latInput) latInput.value = lat;
            if(lonInput) lonInput.value = lon;
        } else {
            console.warn(`Could not geocode '${location}'`);
            if(latInput) latInput.value = '';
            if(lonInput) lonInput.value = '';
        }
    } catch (error) {
        console.error('Geocoding error:', error);
        if(latInput) latInput.value = '';
        if(lonInput) lonInput.value = '';
    }
}

let candleInitialized = false;

export function openProfileModal() {
    const profileOverlay = document.getElementById('profileOverlay');
    profileOverlay.classList.add('active');
    fetchUserProfile();

    if (!candleInitialized) {
        setTimeout(() => {
            initCandle('profileCandleCanvas');
            candleInitialized = true;
        }, 100);
    }
}

export function closeProfileModal() {
    const profileOverlay = document.getElementById('profileOverlay');
    profileOverlay.classList.remove('active');
}

export async function getProfileData(token) {
    if (!token) return null;

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

            let needsRefresh = response.status === 401;
            let errorData = null;

            if (response.status === 400) {
                try {
                    errorData = await response.clone().json();
                    if (errorData && errorData.error && errorData.error.includes("Invalid Token")) {
                        needsRefresh = true;
                    }
                } catch(e) { }
            }

            if (needsRefresh && retryCount === 0) {
                console.log("Token expired or invalid (400/401). Attempting refresh...");
                const refreshToken = localStorage.getItem("tinytorch_refresh_token");
                if (!refreshToken) return null;

                const refreshRes = await fetch(`${NETLIFY_URL}/api/auth/refresh`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ refreshToken })
                });

                if (!refreshRes.ok) return null;

                const refreshData = await refreshRes.json();
                const session = refreshData.session || refreshData;

                if (session && session.access_token) {
                    token = session.access_token;
                    localStorage.setItem("tinytorch_token", token);
                    if (session.refresh_token) {
                        localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
                    }
                    retryCount++;
                    continue;
                } else {
                    return null;
                }
            }

            if (!response.ok) {
                 throw new Error(`Failed to fetch profile data: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error("Error fetching user profile:", error);
            if (retryCount >= MAX_RETRIES) return null;
        }
    } while (retryCount < MAX_RETRIES);
    return null;
}

export async function fetchUserProfile() {
    let token = localStorage.getItem("tinytorch_token");
    if (!token) {
        console.error("No token found for fetching profile.");
        forceLogin();
        return;
    }

    const profileData = await getProfileData(token);
    
    if (profileData) {
        populateProfileForm(profileData.profile);
    } else {
        alert("Failed to load profile data. Please try again.");
        closeProfileModal();
    }
}

async function updateProfileLocation(token, updates) {
    try {
        const response = await fetch(`${SUPABASE_URL}/update-profile`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updates)
        });
        if (response.ok) {
            console.log("Location auto-updated successfully");
        } else {
            console.warn("Failed to auto-update location");
        }
    } catch (e) {
        console.error("Error auto-updating location:", e);
    }
}

export async function checkAndAutoUpdateLocation() {
    const token = localStorage.getItem("tinytorch_token");
    if (!token) return;

    // Check if we already checked in this session to avoid spamming
    if (sessionStorage.getItem("tinytorch_location_checked")) return;

    try {
        const data = await getProfileData(token);
        if (!data || !data.profile) return;

        const p = data.profile;
        
        const isPlaceholder = p.location && (p.location.includes("Lost at Sea") || p.location.includes("🌊"));

        // If location is missing or placeholder, try to find it
        if (isPlaceholder || (!p.location && (!p.latitude || !p.longitude))) {
            console.log("Location missing or placeholder, attempting auto-detection...");
            
            // Try HTTPS first (ipapi.co)
            let found = false;
            try {
                const res = await fetch('https://ipapi.co/json/');
                if (res.ok) {
                    const geo = await res.json();
                    if (geo.city && geo.latitude && geo.longitude) {
                        const locationStr = `${geo.city}, ${geo.country_name}`;
                        await updateProfileLocation(token, {
                            location: locationStr,
                            latitude: geo.latitude,
                            longitude: geo.longitude
                        });
                        found = true;
                    }
                }
            } catch (e) {
                console.warn("Primary geo lookup failed:", e);
            }

            if (!found) {
                // Fallback (might fail on HTTPS sites due to mixed content if using http)
                 try {
                    const res = await fetch('http://ip-api.com/json/');
                    if (res.ok) {
                        const geo = await res.json();
                        if (geo.city && geo.lat && geo.lon) {
                            const locationStr = `${geo.city}, ${geo.country}`;
                            await updateProfileLocation(token, {
                                location: locationStr,
                                latitude: geo.lat,
                                longitude: geo.lon
                            });
                        }
                    }
                } catch (e) {
                     console.warn("Secondary geo lookup failed:", e);
                }
            }
        }
        // Mark as checked for this session
        sessionStorage.setItem("tinytorch_location_checked", "true");
    } catch (e) {
        console.error("Auto-location check failed:", e);
    }
}

function populateProfileForm(data) {
    const profileDisplayNameInput = document.getElementById('profileDisplayName');
    const profileIsPublicCheckbox = document.getElementById('profileIsPublic');
    const profileFullNameInput = document.getElementById('profileFullName');
    const profileSummaryTextarea = document.getElementById('profileSummary');
    const profileLocationInput = document.getElementById('profileLocation');
    const profileInstitutionInput = document.getElementById('profileInstitution');
    const profileRoleInput = document.getElementById('profileRole');
    const profileWebsitesInput = document.getElementById('profileWebsites');
    const profileMailingList = document.getElementById('profileMailingList');
    const profileLatitude = document.getElementById('profileLatitude');
    const profileLongitude = document.getElementById('profileLongitude');

    profileDisplayNameInput.value = data.display_name || '';

    profileFullNameInput.value = data.full_name || '';
    profileSummaryTextarea.value = data.bio || data.summary || '';
    profileLocationInput.value = data.location || '';
    if (profileLatitude) profileLatitude.value = data.latitude || '';
    if (profileLongitude) profileLongitude.value = data.longitude || '';
    
    if (profileRoleInput) {
        profileRoleInput.value = data.role || 'student';
    }

    profileInstitutionInput.value = Array.isArray(data.institution) ? data.institution.join(', ') : (data.institution || '');

    const sites = data.website || data.websites;
    profileWebsitesInput.value = Array.isArray(sites) ? sites.join(', ') : (sites || '');

    if (profileMailingList && data.mailing_list !== undefined && data.mailing_list !== null) {
        profileMailingList.checked = data.mailing_list;
    }
}

export async function handleProfileUpdate(e) {
    e.preventDefault();
    let token = localStorage.getItem("tinytorch_token");
    if (!token) {
        console.error("No token found for updating profile.");
        forceLogin();
        return;
    }

    const profileDisplayNameInput = document.getElementById('profileDisplayName');
    const profileFullNameInput = document.getElementById('profileFullName');
    const profileSummaryTextarea = document.getElementById('profileSummary');
    const profileLocationInput = document.getElementById('profileLocation');
    const profileRoleInput = document.getElementById('profileRole');
    const profileInstitutionInput = document.getElementById('profileInstitution');
    const profileWebsitesInput = document.getElementById('profileWebsites');
    const profileMailingList = document.getElementById('profileMailingList');
    const profileLatitude = document.getElementById('profileLatitude');
    const profileLongitude = document.getElementById('profileLongitude');

    const updatedProfile = {
        display_name: profileDisplayNameInput.value,
        full_name: profileFullNameInput.value,
        summary: profileSummaryTextarea.value,
        role: profileRoleInput ? profileRoleInput.value : 'student',
        location: profileLocationInput.value,
        institution: profileInstitutionInput.value.split(',').map(s => s.trim()).filter(s => s),
        website: profileWebsitesInput.value.split(',').map(s => s.trim()).filter(s => s),
        mailing_list: profileMailingList ? profileMailingList.checked : true,
        latitude: profileLatitude && profileLatitude.value ? parseFloat(profileLatitude.value) : null,
        longitude: profileLongitude && profileLongitude.value ? parseFloat(profileLongitude.value) : null,
    };

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

                const refreshRes = await fetch(`${NETLIFY_URL}/api/auth/refresh`,
                    {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ refreshToken })
                });

                if (!refreshRes.ok) { forceLogin(); return; }

                const refreshData = await refreshRes.json();
                const session = refreshData.session || refreshData;

                if (session && session.access_token) {
                    token = session.access_token;
                    localStorage.setItem("tinytorch_token", token);
                    if (session.refresh_token) {
                        localStorage.setItem("tinytorch_refresh_token", session.refresh_token);
                    }
                    retryCount++;
                    continue;
                } else {
                    console.warn("Refresh failed: No access token in response", refreshData);
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

            // Check if 'community' param is present in the URL
            const params = new URLSearchParams(window.location.search);
            if (params.get('community')) {
                window.location.href = getBasePath() + '/community.html';
            }
            return;
        } catch (error) {
            console.error("Error updating user profile:", error);
            alert("Failed to update profile: " + error.message);
            return;
        }
    } while (retryCount < MAX_RETRIES);
}

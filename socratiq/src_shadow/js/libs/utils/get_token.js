import { BASEURL } from '../../../configs/env_configs'
const TEMPURL = `http://localhost:3000/`
// import { saveToken } from "./utils";
export async function get_token(maxRetries = 10, delayMs = 1000) {
    let attempt = 0;
    
    while (attempt < maxRetries) {
        try {
            // TRACE: Track which AI endpoint is being called
            console.trace(`[AI_TRACE] get_token.js - Calling token endpoint: ${BASEURL}token`, {
              attempt: attempt + 1,
              maxRetries
            });

            const response = await fetch(BASEURL + "token", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username: "username",
                    password: "password",
                }),
            });

            if (!response.ok) {
                throw new Error("Network response was not ok");
            }

            const data = await response.json();
            const token = data.token;
            localStorage.setItem("token_avaya", token);
            return token;

        } catch (error) {
            attempt++;
            console.error(`Token fetch attempt ${attempt}/${maxRetries} failed:`, error);
            
            if (attempt === maxRetries) {
                const fakeToken = "fake_token_12345";
                localStorage.setItem("token_avaya", fakeToken);
                return fakeToken;
            }
            
            // Wait before next retry
            await new Promise(resolve => setTimeout(resolve, delayMs));
        }
    }
}

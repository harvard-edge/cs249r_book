
export function show_token_success(shadowRoot, token) {

    if (token) {
      // Show the success notice
      const successNotice = shadowRoot.getElementById("success-notice");
      successNotice.style.display = "block";
  
      // Hide the success notice after 3 seconds
      setTimeout(() => {
        successNotice.style.display = "none";
      }, 3000);
      return token;
    } else {
      console.error("Error:", "no token found");
  
      // Show the error notice
      const errorNotice = shadowRoot.getElementById("error-notice");
      errorNotice.style.display = "block";
  
      // Hide the error notice after 3 seconds
      setTimeout(() => {
        errorNotice.style.display = "none";
      }, 3000);
    }
  }
  

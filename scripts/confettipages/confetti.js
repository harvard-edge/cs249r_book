// custom.js

document.addEventListener("DOMContentLoaded", function() {
  // Check if it's the user's first visit
  if (!localStorage.getItem('firstVisit')) {
    // Set the first visit flag in local storage
    localStorage.setItem('firstVisit', 'true');

    // Display a welcome message
    const welcomeMessage = document.createElement('div');
    welcomeMessage.innerHTML = `
      <div style="color: white; font-size: 3em; font-weight: bold;">
        Welcome to the ML Systems Book!
      </div>
      <div style="color: white; font-size: 1.5em; margin-top: 20px;">
        A resource for learning about applied machine learning.
      </div>
      <div style="color: white; font-size: 1.2em; margin-top: 30px;">
        Prof. Vijay Janapa Reddi, Harvard University.
      </div>
    `;
    welcomeMessage.style.position = 'fixed';
    welcomeMessage.style.top = '0';
    welcomeMessage.style.left = '0';
    welcomeMessage.style.width = '100%';
    welcomeMessage.style.height = '100%';
    welcomeMessage.style.display = 'flex';
    welcomeMessage.style.flexDirection = 'column';
    welcomeMessage.style.justifyContent = 'center';
    welcomeMessage.style.alignItems = 'center';
    welcomeMessage.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'; // Semi-opaque background
    welcomeMessage.style.color = 'white';
    welcomeMessage.style.textAlign = 'center';
    welcomeMessage.style.zIndex = '1000';
    welcomeMessage.style.opacity = '1';
    welcomeMessage.style.transition = 'opacity 12s'; // Add transition for opacity
    document.body.appendChild(welcomeMessage);

    // Trigger the confetti script
    const script = document.createElement('script');
    script.src = "https://run.confettipage.com/here.js";
    script.setAttribute('data-confetticode', 'U2FsdGVkX1/UZXxxbIcynlCbY0mzzQusg5WULngfD5adgMzEc15y5e1S74UPrrRdk+BQEOXVkrEOaVkGoXEK22dgc2CsHK+KagYyrEv4CE+sbJGfBa6ompBnctYY56wo30NujuwPT7TzKCtV4F4uueEsN8UTuH5STPnHbG9ASOjPyyImEjSYG89SEwGoUw14YnvLuE3QXWpAVRlhh7qUtjiUAidd2bSlTpQmd1OfyFm8pLrO6183KxmLexAxcNFJKh0QkFrxG/LEAA+4vIIZrL4gATrNkcUrbdeL2VDsQGAOPFXJVzA50fZUtiawk+FbUD9kIhvTEOx9XzwwIq5WRlnbltT0uz4WZHSqZ2cdvHFLzpV+HqkQm76LPDLCgyA4Pbo2EbDTlXpFIx8BuwTx7H2idFrDO9zLgPAlrj0g+h4NznQV+B55vEGQiryEudyihPRzraPAT5vKpYbr7k62jV5msRO2O1pG+2HmH7e9z5v3+74cBluP1qrn52OZXdVQlmYuXRG2/kpMwDClhHxIGHzi3AWn1zH+sAJ+ICtXuZu02L+hTKkdpr/OHu4jMoz3F5vpOusFXLihx6byza+BukY2MDbTjFNICgZSm6JTbTisQWB1oEcevwOewKjoWybeQiPmEtKkXmQlBp3liYCKwKKKsyt4JnnoMZrczo0fTB8=');
    document.body.appendChild(script);

    // Fade out the welcome message after a few seconds
    setTimeout(() => {
      welcomeMessage.style.opacity = '0';
    }, 3000); // Adjust the delay before fading out as needed

    // Remove the welcome message from the DOM after the fade-out transition
    setTimeout(() => {
      welcomeMessage.remove();
    }, 5000); // This should be slightly longer than the fade-out transition duration
  }
});

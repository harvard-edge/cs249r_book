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
    script.setAttribute('data-confetticode', 'U2FsdGVkX18PTb8Vsl0OaWt5DlYvi1EpmZ1g03dN0/He6MHxX9/qmVN93W7pVyq4VCVNYVctaZocBF/2m59tpKmgNiqPF8sS/3l8DqM91vBrhexmAFNHStQnYLxc4bNbBhbY537N2MVg6oWhKkm4OrYXHf0sE3fYXuuO7lp8folqy0rnmXBJKEHYZSJdusaOs7AmxgFOJqkqSViJHqpvkzE5dc2dGkrqr4OAF7lz8OaAk9brMe6f3irUFRuiyaziBkJn+m1nz0LO6Wqq9QC/lxd9bJPKD9k8Ra1sM9uYnFpVcuyQgHb04IK7HBnDIW4FkZc30x9zmUBgtiY0KQU6myHvDXJld+5r0RwqCKUV9DB79Bulw6UjtbKEwhNGC16AtNEiEl1QQjMI7ml9stj9/IOcnSgBYxtO+6UPan9BVKswi0ZZeNo7lrzjDU9o2IoMMlagMfi7XvbVLcAXJ+lkAyOnRsgCxkJSmPYff9FuBLLht5HEQ33Nlj4DKBx9YSoH28l4BEkJkoQ8rWSJlEZTet7Of0WI7bZ+WKUsvUHnvBt3pO9reNSxRP1SEFIKjHSYL7jYjFwZm21WhMsZ0wNmPCaXsHw3r6XtTL4w8KFbkiTEt+U/naTUFsumcPdKSolk8gmGDSD2aQ73askV71vrcvIHKyxs8UWOYaxGznj1VO9i97T/784QAll6d/9+ymHZrGTH1UbSz3KRU9uGoBqtFw==');
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

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
        A collaborative resource for learning about machine learning systems.
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
    script.setAttribute('data-confetticode', 'U2FsdGVkX18E6TB2Ivq4bDNL8dgJIDqYuJNnkw3i+vLHSP+0Ux9g2Wggjp7GTpcoDoBYY2vUrMX6sFtotQTU4ui2NTwXGD639xEvpHkfNuoa7dpCWwLDKnqv6ZP6hwn2pYTUScU9NrMKvYMybq73VRLrRKhNrEDzlZfKVYLqsV64WfkmBRMm6uu+dV7UvveoOelfZau3uLgD+hBaD2DcbEiGndBMmAWbtIpWX8P5+lV9CznNqPYYgxIf4vM7zbTqBK7y+qXlrPx1kkS826myGs/p3rKR8haApxKNwOhbx8IDGsauUQ0iARO3tLAeMP/0B+IvwJJYoCbSai+/l2CXNSCIOmkoEt8IOD86u9XzZpvSWQT3Pu0aRDk2zerp+BsqEYQQmoJYdjbGGrDfQ8ste0x1hYTpvYFIydq9cBUSd0wiuMBsJ5iY9YamkrYuAoHgo0GBClOTXMSek4VgRSbNUkAQt8Gc/KLyhKrkMCuv+a1YiDazuHEZ8T9x70YLI1r52Kd/z4MP4ROCgqrpI17CnjyOs3Dk7pj1/i93Hd6U56W/E/6eMn2EF8IQFPaUmmmulnGD62UhzcgRF+c9fbO/oTtoCH0804Nmm23iaxWjq7rVZAzteziBgBF25ACDPjFwxXHESnYfDs3MWyquqwcLSR5j9SuMlF97reG+g4qXJdbVr5gnR/RaVMXAhd8+yZkv');
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

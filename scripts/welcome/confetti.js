// custom.js

document.addEventListener("DOMContentLoaded", function() {
  // Check if it's the user's first visit
  if (!localStorage.getItem('firstVisit')) {
    // Set the first visit flag in local storage
    localStorage.setItem('firstVisit', 'false');

    // Display a welcome message
    const welcomeMessage = document.createElement('div');
    welcomeMessage.innerText = "Welcome!";
    welcomeMessage.style.position = 'fixed';
    welcomeMessage.style.top = '0';
    welcomeMessage.style.left = '0';
    welcomeMessage.style.width = '100%';
    welcomeMessage.style.height = '100%';
    welcomeMessage.style.display = 'flex';
    welcomeMessage.style.justifyContent = 'center';
    welcomeMessage.style.alignItems = 'center';
    welcomeMessage.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'; // Semi-opaque background
    welcomeMessage.style.color = 'white';
    welcomeMessage.style.fontSize = '5em';
    welcomeMessage.style.textAlign = 'center';
    welcomeMessage.style.zIndex = '1000';
    welcomeMessage.style.opacity = '1';
    welcomeMessage.style.transition = 'opacity 2s'; // Add transition for opacity
    document.body.appendChild(welcomeMessage);

    // Trigger the confetti script
    const script = document.createElement('script');
    script.src = "https://run.confettipage.com/here.js";
    script.setAttribute('data-confetticode', 'U2FsdGVkX19H2OWLrooaDFegLFthmLw91OrLhii6zjU20bEELdqOM8ZxlWHaher3attSQqQNveav1dCC3XESFbiKKL3qGa6sokY4qQSp0V72daxUsoizkPE0l0tfsnKYakUooHoXs8GphMowW0j0N8ahebqVKyR0Nrs6BfXMxZdhqbEfJeXE8q9bR6n5RSzwqsqwPlFfzHwRR8idNCDzPkMVk88SSlD5lDgi270KNv0tBLZw8xPBktSTUnA9itFgT/TgaLDaDjk+9lOn8wKzg3BdDcVo3dFTpQ3q0mDSYhNe2tu0q1H2r4WQfgou5EFy273XQMjjTrO8KBRtZmDHR0xixnuCT9bWo9C9uwKTXu8a2sXCDx0q5K3ja9VK7dBY0YcE4hTUK0J5yHWY4/WqOTvgj0EKz+qd4hxIXKVPYZdikAYyfEvGDoFydZELvHvg9P0uNKz5bKZ8VhSi5GkVPdrCyjOexM3zLvS2XQwleGfMYkt9ik0OsDkUeb8W4cDT81LxZibyN55+U6PUrzeWjCdP65BEoIWZuYkXLLxPkl9XQOe4nBRo6CDHa05Wx8atl4cDbIwcEWVckPDTjbgFPRKYAA663I0sPIRiQh1ZYc6/kEUUmEFwWKo4L6tltB3oMa8AlQIZu6I6Tih/DoU2fcfJ5YanCpDJmv2zOX+g1a8IUuslOrMdnRzEYXVhgVjU');
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

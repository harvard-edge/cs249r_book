import main from 'boarding.js/styles/main.css?inline';
import basic from 'boarding.js/styles/themes/basic.css?inline';

export const boardingStyles = `${main}\n${basic}`;

// Boarding mounts its overlay outside the widget's shadow root.
export function ensureBoardingStyles() {
  if (document.getElementById('socratiq-boarding-styles')) return;
  const style = document.createElement('style');
  style.id = 'socratiq-boarding-styles';
  style.textContent = boardingStyles;
  document.head.appendChild(style);
}

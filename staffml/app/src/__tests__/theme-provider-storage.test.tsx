/**
 * ThemeProvider storage-guard regression.
 *
 * getInitialTheme() reads localStorage to reconcile the stored theme on mount.
 * In strict-storage browsers (Safari "Block all cookies", sandboxed iframes)
 * localStorage.getItem itself throws. The writer (toggleTheme) already guarded
 * its setItem calls, but the reader did not — so the mount effect threw and the
 * theme never reconciled. The read is now wrapped in try/catch, falling back to
 * the ecosystem default ("light").
 */
import { afterEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import ThemeProvider, { useTheme } from '@/components/ThemeProvider';

function ThemeReadout() {
  const { theme } = useTheme();
  return <span data-testid="theme">{theme}</span>;
}

const originalGetItem = window.localStorage.getItem;

afterEach(() => {
  window.localStorage.getItem = originalGetItem;
});

describe('ThemeProvider storage guard', () => {
  it('falls back to the default theme without throwing when localStorage.getItem throws', () => {
    window.localStorage.getItem = () => {
      throw new DOMException('The operation is insecure.', 'SecurityError');
    };

    expect(() =>
      render(
        <ThemeProvider>
          <ThemeReadout />
        </ThemeProvider>,
      ),
    ).not.toThrow();

    expect(screen.getByTestId('theme')).toHaveTextContent('light');
  });

  it('reconciles to a stored theme when storage is readable', () => {
    window.localStorage.setItem('staffml_theme', 'dark');

    render(
      <ThemeProvider>
        <ThemeReadout />
      </ThemeProvider>,
    );

    expect(screen.getByTestId('theme')).toHaveTextContent('dark');
  });
});

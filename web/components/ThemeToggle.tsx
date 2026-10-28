'use client';

import { useEffect, useState } from 'react';
import { resolveTheme, THEME_STORAGE_KEY, toggleTheme, type Theme } from '@/lib/theme';

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('light');

  useEffect(() => {
    let savedTheme: Theme = 'light';
    try {
      savedTheme = resolveTheme(localStorage.getItem(THEME_STORAGE_KEY));
    } catch {
      // Storage can be unavailable; the toggle still works for this page.
    }
    document.documentElement.setAttribute('data-theme', savedTheme);
    // The server cannot read browser storage; synchronize once after mounting.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTheme(savedTheme);
  }, []);

  function handleToggle() {
    const nextTheme = toggleTheme(theme);
    setTheme(nextTheme);
    document.documentElement.setAttribute('data-theme', nextTheme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch {
      // Keep the selected theme usable when persistence is blocked.
    }
  }

  const label = theme === 'light' ? 'Light' : 'Dark';
  const description = `${label} mode. Switch to ${toggleTheme(theme)} mode`;

  return (
    <button
      type="button"
      onClick={handleToggle}
      aria-label={description}
      title={description}
      className="inline-flex min-h-11 shrink-0 items-center gap-2 rounded-control border border-line bg-card px-3 py-2 text-sm font-bold text-ink transition hover:border-action"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {theme === 'light' ? (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" />
          </>
        ) : (
          <path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z" />
        )}
      </svg>
      {label}
    </button>
  );
}

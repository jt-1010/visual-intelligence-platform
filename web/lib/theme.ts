export const THEME_STORAGE_KEY = 'theme';

export type Theme = 'light' | 'dark';

export function resolveTheme(value: string | null): Theme {
  return value === 'dark' ? 'dark' : 'light';
}

export function toggleTheme(theme: Theme): Theme {
  return theme === 'light' ? 'dark' : 'light';
}

// Run before the page paints, using the same validation as the toggle.
export const THEME_INIT_SCRIPT = `(() => {
  let theme = 'light';
  try {
    const storedTheme = localStorage.getItem('${THEME_STORAGE_KEY}');
    theme = storedTheme === 'dark' ? 'dark' : 'light';
  } catch {}
  document.documentElement.setAttribute('data-theme', theme);
})();`;

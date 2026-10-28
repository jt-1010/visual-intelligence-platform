import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { resolveTheme, THEME_INIT_SCRIPT, THEME_STORAGE_KEY, toggleTheme } from '@/lib/theme';

describe('theme preferences', () => {
  it('uses the theme storage key', () => {
    assert.equal(THEME_STORAGE_KEY, 'theme');
  });

  it('accepts saved themes and defaults missing or invalid values to light', () => {
    assert.equal(resolveTheme('dark'), 'dark');
    for (const value of [null, '', 'light', 'Dark', 'system', 'invalid']) {
      assert.equal(resolveTheme(value), 'light');
    }
  });

  it('toggles in both directions', () => {
    assert.equal(toggleTheme('light'), 'dark');
    assert.equal(toggleTheme('dark'), 'light');
  });

  it('restores only valid saved preferences before painting', () => {
    for (const storedTheme of [null, 'light', 'dark', 'invalid']) {
      const attributes = new Map<string, string>();
      runInNewContext(THEME_INIT_SCRIPT, {
        localStorage: {
          getItem(key: string) {
            assert.equal(key, 'theme');
            return storedTheme;
          },
        },
        document: { documentElement: { setAttribute: (key: string, value: string) => attributes.set(key, value) } },
      });
      assert.equal(attributes.get('data-theme'), resolveTheme(storedTheme));
    }
  });

  it('keeps light mode when storage access throws', () => {
    const attributes = new Map<string, string>();
    runInNewContext(THEME_INIT_SCRIPT, {
      localStorage: { getItem() { throw new Error('Storage blocked'); } },
      document: { documentElement: { setAttribute: (key: string, value: string) => attributes.set(key, value) } },
    });
    assert.equal(attributes.get('data-theme'), 'light');
  });
});

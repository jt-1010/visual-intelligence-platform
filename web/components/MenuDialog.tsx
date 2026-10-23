'use client';

import { useEffect, useRef, useState } from 'react';
import { MenuGrid, type MenuItem } from './MenuGrid';

type Props = {
  open: boolean;
  onClose: () => void;
  categories: Record<string, MenuItem[]>;
  onPick: (item: MenuItem) => void;
  disabled: boolean;
};

/**
 * The menu as a popup, with a button to take it full screen.
 *
 * The menu used to be a permanent sidebar, which took a third of the screen
 * away from the conversation even when nobody was browsing. As a popup it
 * gets the whole width when it is wanted and none of it when it is not.
 *
 * Built on <dialog> with showModal(), so focus is trapped inside while it is
 * open and Escape closes it, with no focus-management code of our own.
 */
export function MenuDialog({ open, onClose, categories, onPick, disabled }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      aria-label="Menu"
      className={[
        'm-auto flex-col overflow-hidden border-2 border-line-strong bg-card p-0 text-ink open:flex',
        'backdrop:bg-sunk',
        fullscreen
          ? 'h-dvh max-h-none w-dvw max-w-none rounded-none'
          : 'h-[85dvh] w-[min(64rem,94vw)] rounded-panel',
      ].join(' ')}
    >
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-6 py-4">
        <h2 className="text-[1.375rem] font-bold tracking-[-0.01em]">Menu</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setFullscreen((f) => !f)}
            aria-pressed={fullscreen}
            className="min-h-11 rounded-control border border-line-strong bg-card px-4 text-[1rem] font-bold text-ink transition hover:bg-sunk"
          >
            {fullscreen ? 'Exit full screen' : 'Full screen'}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="min-h-11 rounded-control border border-line-strong bg-card px-4 text-[1rem] font-bold text-ink transition hover:bg-sunk"
          >
            Close
          </button>
        </div>
      </div>

      <div className="@container min-h-0 flex-1 overflow-y-auto px-6 py-5">
        <MenuGrid
          categories={categories}
          disabled={disabled}
          onPick={(item) => {
            onPick(item);
            onClose();
          }}
        />
      </div>
    </dialog>
  );
}

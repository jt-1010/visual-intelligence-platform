'use client';

import { formatMoney } from '@/lib/money';

export type MenuItem = {
  id: number;
  slug: string;
  name: string;
  description: string;
  category: string;
  priceCents: number;
  calories: number | null;
  imageUrl: string | null;
};

type Props = {
  categories: Record<string, MenuItem[]>;
  onPick: (item: MenuItem) => void;
  disabled?: boolean;
};

const CATEGORY_LABEL: Record<string, string> = {
  burgers: 'Burgers',
  chicken: 'Chicken',
  sides: 'Sides',
  drinks: 'Drinks',
  desserts: 'Desserts',
  salads: 'Salads',
  breakfast: 'Breakfast',
  combos: 'Meals',
  other: 'More',
};

/**
 * Touch ordering, routed through the same agent as signing and speech.
 *
 * Tapping does not change the cart directly -- it sends a message and lets the
 * agent act on it. That keeps one conversation, so the read-back at the end
 * covers everything however it was ordered, and someone can start by tapping
 * and finish by signing without the terminal losing the thread.
 */
export function MenuGrid({ categories, onPick, disabled }: Props) {
  const names = Object.keys(categories);
  if (names.length === 0) return null;

  return (
    <div className="space-y-7">
      {names.map((category) => (
        <section key={category}>
          <h3 className="mb-3 text-[1.125rem] font-bold tracking-[-0.01em]">
            {CATEGORY_LABEL[category] ?? category}
          </h3>
          {/*
            One item per row, not a grid.

            This panel is a fixed ~23rem column, and Tailwind's responsive
            prefixes key off the VIEWPORT, not the container -- so `lg:grid-cols-3`
            kicked in on a wide screen precisely where the sidebar could least
            afford it. Cards came out 104px wide: a 56px photo, padding, and
            sixteen pixels of room for the name, which rendered every item as
            "B.." or "C..". A row gives the name the width it needs and makes
            the price easy to scan down the right edge.
          */}
          <ul className="space-y-2">
            {categories[category].map((item) => (
              <li key={item.slug}>
                <button
                  type="button"
                  onClick={() => onPick(item)}
                  disabled={disabled}
                  className="flex w-full items-center gap-3.5 rounded-control border border-line bg-card p-3 text-left transition hover:border-action hover:bg-action-soft focus-visible:border-action disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {item.imageUrl && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={item.imageUrl}
                      alt=""
                      aria-hidden="true"
                      className="h-14 w-14 shrink-0 rounded-[0.4rem] object-cover"
                    />
                  )}
                  <span className="min-w-0 flex-1">
                    <span className="block text-[1.0625rem] font-bold leading-snug">
                      {item.name}
                    </span>
                    {item.calories !== null && (
                      <span className="tnum mt-0.5 block text-[0.9375rem] text-ink-soft">
                        {item.calories} cal
                      </span>
                    )}
                  </span>
                  <span className="tnum shrink-0 text-[1.0625rem] font-bold">
                    {formatMoney(item.priceCents)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

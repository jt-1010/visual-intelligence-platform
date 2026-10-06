import { tool, type ToolSet } from 'ai';
import { z } from 'zod';
import {
  addToCart,
  clearCart,
  confirmOrder,
  formatMoney,
  getCart,
  removeFromCart,
  searchMenu,
  suggestAlternatives,
} from './cart';
import { getRecommendations } from './recommend';
import { getDb } from '@/lib/db';
import { menuItems } from '@/lib/db/schema';
import { inArray } from 'drizzle-orm';

/**
 * Every tool returns prices as BOTH cents and a preformatted string.
 *
 * The model is told to read prices back verbatim, and handing it
 * "$5.99" removes the one step where it could do arithmetic and be wrong.
 * The cents field is what the UI and the tests actually assert on.
 */

export function buildTools(sessionId: string): ToolSet {
  return {
    search_menu: tool({
      description:
        'Search the menu. Use this before adding anything, and whenever the person asks what is available. Returns real items with real prices.',
      inputSchema: z.object({
        query: z
          .string()
          .describe('What to look for: an item name, a food word, or a category like "drinks".'),
      }),
      execute: async ({ query }) => {
        const results = await searchMenu(query, 5);

        // Never return an empty answer. If we sell nothing like it, say so and
        // hand back the nearest things we do sell, so the assistant can offer
        // an alternative in the same breath instead of a flat refusal.
        if (results.length === 0) {
          const alternatives = await suggestAlternatives(query, 3);
          return {
            count: 0,
            weDoNotSellThis: true,
            message: `Nothing on the menu matches "${query}".`,
            closestWeDoHave: alternatives.map((a) => ({
              name: a.name,
              category: a.category,
              price: formatMoney(a.priceCents),
            })),
          };
        }

        return {
          count: results.length,
          items: results.map((r) => ({
            slug: r.slug,
            name: r.name,
            description: r.description,
            category: r.category,
            price: formatMoney(r.priceCents),
            priceCents: r.priceCents,
            calories: r.calories,
          })),
        };
      },
    }),

    add_to_cart: tool({
      description:
        'Add an item to the order. If the item cannot be matched confidently this returns suggestions instead of guessing - offer those to the person.',
      inputSchema: z.object({
        item: z.string().describe('The item to add, as the person referred to it.'),
        quantity: z.number().int().min(1).max(20).default(1),
        modifiers: z
          .array(z.string())
          .default([])
          .describe('Customisations such as "no onion", "extra cheese", "no ice".'),
      }),
      execute: async ({ item, quantity, modifiers }) => {
        const result = await addToCart(sessionId, item, quantity, modifiers);

        // The person named a KIND of thing, not a thing. Nothing was added.
        // The options below are the real menu rows -- offer them verbatim and
        // do not pick one on the person's behalf.
        if (!result.ok && result.reason === 'ambiguous') {
          return {
            added: false,
            needsChoice: true,
            askedFor: result.query,
            message: `"${result.query}" matches ${result.totalMatches} items. Ask which one.`,
            totalMatches: result.totalMatches,
            options: result.options.map((o) => ({
              name: o.name,
              price: formatMoney(o.priceCents),
              calories: o.calories,
            })),
            quantityTheyWanted: quantity,
          };
        }

        if (!result.ok) {
          return {
            added: false,
            message: `No confident match for "${item}".`,
            suggestions: result.suggestions.map((s) => ({
              name: s.name,
              price: formatMoney(s.priceCents),
            })),
          };
        }

        return {
          added: true,
          item: {
            name: result.added.name,
            quantity: result.added.qty,
            unitPrice: formatMoney(result.added.unitPriceCents),
            modifiers: result.added.modifiers.map((m) => m.name),
            lineTotal: formatMoney(result.added.lineTotalCents),
          },
          orderTotal: formatMoney(result.cart.totalCents),
        };
      },
    }),

    remove_from_cart: tool({
      description: 'Remove an item from the order.',
      inputSchema: z.object({
        item: z.string().describe('The item to remove, as the person referred to it.'),
      }),
      execute: async ({ item }) => {
        const result = await removeFromCart(sessionId, item);
        return result.ok
          ? {
              removed: true,
              item: result.removed!.name,
              orderTotal: formatMoney(result.cart.totalCents),
            }
          : {
              removed: false,
              message: `"${item}" is not in the order.`,
              currentItems: result.cart.lines.map((l) => l.name),
            };
      },
    }),

    get_cart: tool({
      description:
        'Read the current order back. Call this before summarising or confirming - never total the items yourself.',
      inputSchema: z.object({}),
      execute: async () => {
        const cart = await getCart(sessionId);
        return {
          empty: cart.lines.length === 0,
          items: cart.lines.map((l) => ({
            name: l.name,
            quantity: l.qty,
            modifiers: l.modifiers.map((m) => m.name),
            lineTotal: formatMoney(l.lineTotalCents),
          })),
          subtotal: formatMoney(cart.subtotalCents),
          tax: formatMoney(cart.taxCents),
          total: formatMoney(cart.totalCents),
        };
      },
    }),

    get_recommendations: tool({
      description:
        'Get at most one or two suggested add-ons for the current order. Use this once, when the order looks nearly complete. Never suggest the same thing twice.',
      inputSchema: z.object({}),
      execute: async () => {
        const cart = await getCart(sessionId);
        if (cart.lines.length === 0) return { recommendations: [] };

        const db = await getDb();
        const slugs = cart.lines.map((l) => l.slug);
        const rows = await db
          .select({ category: menuItems.category })
          .from(menuItems)
          .where(inArray(menuItems.slug, slugs));

        const recs = await getRecommendations(
          slugs,
          rows.map((r) => r.category),
          2,
        );

        return {
          recommendations: recs.map((r) => ({
            name: r.name,
            price: formatMoney(r.priceCents),
            reason: r.reason,
          })),
        };
      },
    }),

    confirm_order: tool({
      description:
        'Finalise the order. Only call this after the person has clearly agreed to the read-back total.',
      inputSchema: z.object({}),
      execute: async () => {
        const result = await confirmOrder(sessionId);
        return result.ok
          ? {
              confirmed: true,
              orderNumber: result.orderId,
              total: formatMoney(result.cart.totalCents),
            }
          : { confirmed: false, message: 'The order is empty - nothing to confirm.' };
      },
    }),

    clear_order: tool({
      description: 'Empty the order and start over. Use when the person asks to cancel or restart.',
      inputSchema: z.object({}),
      execute: async () => {
        await clearCart(sessionId);
        return { cleared: true };
      },
    }),
  };
}

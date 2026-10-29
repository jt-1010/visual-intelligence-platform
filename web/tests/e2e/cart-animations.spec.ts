import { expect, test, type Page, type WebSocketRoute } from '@playwright/test';
import path from 'node:path';

async function order(page: Page, text = '2 Hamburger') {
  await page.getByRole('textbox', { name: 'Type your order' }).fill(text);
  await page.getByRole('button', { name: 'Send', exact: true }).click();
}

async function addItems(page: Page) {
  await page.goto('/asl');
  await order(page);
  const item = page.getByRole('complementary', { name: 'Your order', exact: true }).locator('li').first();
  await expect(item).toContainText('Hamburger');
  await expect(page.getByRole('button', { name: 'Confirm order', exact: true })).toBeEnabled();
  return item;
}

test('adding items animates cart entries from offset to their final position', async ({ page }) => {
  const item = await addItems(page);
  await expect(item).toHaveClass(/cart-line-item/);
  const frames = await item.evaluate((el) => {
    const animation = el.getAnimations()[0];
    animation.pause();
    animation.currentTime = 0;
    const start = { opacity: getComputedStyle(el).opacity, transform: getComputedStyle(el).transform };
    animation.currentTime = 360;
    const end = { opacity: getComputedStyle(el).opacity, transform: getComputedStyle(el).transform };
    return { start, end };
  });
  expect(frames.start.opacity).toBe('0');
  expect(frames.start.transform).toContain('14');
  expect(frames.end.opacity).toBe('1');
  expect(frames.end.transform).toBe('matrix(1, 0, 0, 1, 0, 0)');
});

test('confirmation animates the badge and checkmark, shows a receipt, and starts a fresh order', async ({ page }) => {
  let signSocket: WebSocketRoute;
  await page.routeWebSocket('**/ws/sign', (socket) => { signSocket = socket; });
  await addItems(page);
  await page.getByRole('button', { name: 'Confirm order', exact: true }).click();
  const panel = page.getByRole('complementary', { name: 'Order confirmation' });
  await expect(panel.getByRole('heading', { name: 'Thanks for ordering!' })).toBeVisible();
  await expect(panel.getByText(/Order #\d+/)).toBeVisible();
  await expect(panel.getByText('No payment needed for demo')).toBeVisible();
  await expect(panel.getByRole('region', { name: 'Receipt summary' })).toContainText('Hamburger');
  await expect(panel.locator('li')).toContainText('2 ×');
  await expect(panel.getByRole('status')).toHaveText('Sent to kitchen display');
  const animationNames = await panel.evaluate((el) => el.getAnimations({ subtree: true }).map((a) => (a as CSSAnimation).animationName));
  expect(animationNames).toEqual(expect.arrayContaining(['pop-in', 'slide-up', 'pulse-glow', 'check-draw']));
  const check = panel.locator('.check-icon-polyline');
  const offsets = await check.evaluate((el) => {
    const animation = el.getAnimations()[0];
    animation.pause();
    animation.currentTime = 220;
    const start = getComputedStyle(el).strokeDashoffset;
    animation.currentTime = 640;
    return { start, end: getComputedStyle(el).strokeDashoffset };
  });
  expect(offsets).toEqual({ start: '48px', end: '0px' });
  await page.getByRole('button', { name: 'Start a new order' }).click();
  await expect(panel).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Add something first' })).toBeVisible();
  await order(page, 'Hamburger');
  await expect(page.getByRole('button', { name: 'Confirm order', exact: true })).toBeEnabled();
  // Inject a recognised ASL gloss through the actual signing input channel.
  signSocket!.send(JSON.stringify({ type: 'sign', label: 'yes', confidence: 0.99, motion: 1, latencyMs: 5, model: 'test' }));
  await expect(panel).toBeVisible();
});

test('confirmation layout screenshot and reduced motion remain readable', async ({ page }) => {
  await addItems(page);
  await page.getByRole('button', { name: 'Confirm order', exact: true }).click();
  const panel = page.getByRole('complementary', { name: 'Order confirmation' });
  await expect(panel).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await panel.evaluate((el) => el.getAnimations({ subtree: true }).forEach((a) => a.finish()));
  const box = await panel.boundingBox();
  const button = await panel.getByRole('button', { name: 'Start a new order' }).boundingBox();
  expect(box).not.toBeNull();
  expect(button!.y + button!.height).toBeLessThanOrEqual(box!.y + box!.height);
  expect(await panel.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
  await panel.screenshot({ path: path.join(__dirname, 'screenshots/confirmation.png'), animations: 'disabled' });
  await expect(panel).toHaveScreenshot('confirmation-baseline.png', {
    animations: 'disabled', mask: [panel.getByText(/Order #\d+/)], maxDiffPixels: 0,
  });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  expect(await panel.evaluate((el) => el.getAnimations({ subtree: true }).length)).toBe(0);
  await expect(panel.locator('.check-icon-polyline')).toHaveCSS('stroke-dashoffset', '0px');
  await expect(panel.getByRole('heading', { name: 'Thanks for ordering!' })).toBeVisible();
});

/* Browser smoke test. Set PLAYWRIGHT_MODULE and BRAVE_PATH (or CHROME_PATH) for your machine. */
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const output = path.resolve(__dirname, '../tmp/screenshots');
  fs.mkdirSync(output, { recursive: true });
  const executablePath = process.env.BRAVE_PATH || process.env.CHROME_PATH;
  assert.ok(executablePath, 'Set BRAVE_PATH to your Brave executable before running this test');
  assert.match(path.basename(executablePath), /^brave(?:\.exe)?$/i, 'This project is verified with Brave');
  const browser = await chromium.launch({
    headless: true,
    executablePath,
  });
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  page.on('pageerror', error => errors.push(error.message));
  const base = process.env.VERIFUSE_URL || 'http://127.0.0.1:8765';
  let predictions = 0;
  page.on('request', request => { if (request.url().endsWith('/api/predict')) predictions += 1; });
  async function analyze() {
    const response = page.waitForResponse(response => response.url().endsWith('/api/predict'));
    await page.locator('#analyze-button').click();
    assert.equal((await response).status(), 200);
    await page.waitForFunction(() => !document.querySelector('#result-content').hidden, null, { timeout: 20000 });
    await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
  }
  async function releaseRead(index) {
    await page.evaluate(index => new Promise(resolve => {
      const pending = window.__pendingReads[index];
      pending.reader.addEventListener('loadend', resolve, { once: true });
      pending.start();
    }), index);
  }
  try {
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.screenshot({ path: path.join(output, 'verify-desktop.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }' });
    await page.locator('#analyze-button').click();
    await page.waitForFunction(() => !document.querySelector('#form-error').hidden);
    const samples = page.locator('#samples button');
    assert.ok(await samples.count() > 0, 'Sample gallery should load');
    await samples.first().click();
    assert.ok((await page.locator('#content-text').inputValue()).length > 0, 'Sample caption should populate');
    await analyze();
    assert.match(await page.locator('#result-content').innerText(), /REAL|FAKE|UNCERTAIN/);
    assert.match(await page.locator('.result-detail').innerText(), /supported scene domain/);
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, 'prediction-desktop.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }' });

    // Export the submitted snapshot even if code changes the form without firing an input event.
    const originalCaption = await page.locator('#content-text').inputValue();
    await page.evaluate(() => { document.querySelector('#content-text').value = 'Unsubmitted edit'; });
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#download-result').click();
    const download = await downloadPromise;
    const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal(exported.input.text, originalCaption);
    await page.locator('#content-text').fill(originalCaption);
    assert.equal(await page.locator('#result-content').isVisible(), false, 'Editing invalidates the old result');

    // Hold FileReader completion explicitly so competing uploads are deterministic.
    const sampleImage = await page.request.get(new URL(await samples.first().locator('img').getAttribute('src'), base).href);
    const imageBuffer = await sampleImage.body();
    await page.evaluate(() => {
      window.__pendingReads = [];
      window.__originalRead = FileReader.prototype.readAsDataURL;
      FileReader.prototype.readAsDataURL = function(file) {
        window.__pendingReads.push({ reader: this, start: () => window.__originalRead.call(this, file) });
      };
    });
    const upload = name => page.locator('#image-file').setInputFiles({ name, mimeType: 'image/png', buffer: imageBuffer });
    await upload('cancelled-upload.png');
    assert.equal(await page.locator('#analyze-button').isDisabled(), true, 'Analyze waits for image reading');
    const predictionCount = predictions;
    await page.locator('#verify-form').dispatchEvent('submit');
    assert.match(await page.locator('#form-error').innerText(), /finish loading/);
    assert.equal(predictions, predictionCount, 'Pending upload must not submit a stale image');
    await samples.nth(1).click();
    const selectedName = await page.locator('#preview-name').innerText();
    await releaseRead(0);
    assert.equal(await page.locator('#preview-name').innerText(), selectedName, 'Cancelled upload cannot replace a selected sample');
    await upload('older-upload.png');
    await upload('newest-upload.png');
    await releaseRead(2);
    await page.waitForFunction(() => document.querySelector('#preview-name').textContent === 'newest-upload.png');
    await releaseRead(1);
    assert.equal(await page.locator('#preview-name').innerText(), 'newest-upload.png', 'Latest upload wins independently of completion order');
    await page.evaluate(() => { FileReader.prototype.readAsDataURL = window.__originalRead; });
    await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
    await analyze();
    await page.locator('#image-file').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('not a PNG') });
    await page.waitForFunction(() => document.querySelector('#form-error').textContent.includes('readable image'));
    await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
    assert.equal(await page.locator('#preview-name').innerText(), 'newest-upload.png', 'Unreadable upload preserves the previous valid image');
    await page.locator('#remove-image').click();
    await page.locator('#content-text').fill('A news report makes an unsupported claim.');
    await analyze();
    assert.equal(await page.locator('.verdict').innerText(), 'UNCERTAIN');

    // The form must recover from an API error and work on the following request.
    await page.route('**/api/predict', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Test service unavailable' }) }), { times: 1 });
    await page.locator('#analyze-button').click();
    await page.waitForFunction(() => document.querySelector('#form-error').textContent === 'Test service unavailable');
    await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
    assert.equal(await page.locator('#result-status').innerText(), 'ANALYSIS UNAVAILABLE');
    await samples.first().click();
    await page.locator('#model-select').selectOption('late_fusion');
    await analyze();
    assert.match(await page.locator('.result-detail').innerText(), /Late Fusion/);

    await page.locator('[data-view="experiments"]').click();
    await page.waitForFunction(() => document.querySelector('#experiments-content').querySelector('table'));
    const experiments = await (await page.request.get(`${base}/api/experiments`)).json();
    const sanityLabel = `${experiments.sanity_checks.filter(check => check.passed).length} / ${experiments.sanity_checks.length} CHECKS PASSED`;
    assert.ok((await page.locator('#experiments-content').innerText()).includes(sanityLabel), 'Sanity counts follow the actual result schema');
    assert.equal(await page.locator('.case-detail').count(), experiments.difficult_cases.length);
    await page.screenshot({ path: path.join(output, 'experiments-desktop.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }' });
    await page.locator('[data-view="about"]').click();
    assert.equal(await page.locator('.deliverable-links a').count(), 4);
    await page.screenshot({ path: path.join(output, 'about-desktop.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }' });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.screenshot({ path: path.join(output, 'verify-mobile.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }' });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
    assert.equal(overflow, false, 'Mobile view should not overflow horizontally');
    await page.locator('#menu-toggle').click();
    await page.locator('[data-view="about"]').click();
    assert.equal(await page.locator('#menu-toggle').getAttribute('aria-expanded'), 'false');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false, 'Project downloads fit on mobile');
    await page.screenshot({ path: path.join(output, 'about-mobile.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }' });
    assert.deepEqual(errors, [], 'No unhandled browser errors');
    console.log(JSON.stringify({ status: 'passed', browser: 'Brave', predictions, sanityChecks: experiments.sanity_checks.length, screenshots: output, browserErrors: errors }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

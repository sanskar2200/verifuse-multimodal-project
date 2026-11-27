/* Actual real-profile inference and UI check in the user's requested Brave browser. */
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const output = path.join(root, 'tmp/real-ui');
  fs.mkdirSync(output, { recursive: true });
  const executablePath = process.env.BRAVE_PATH;
  assert.match(path.basename(executablePath || ''), /^brave(?:\.exe)?$/i);
  const predictions = JSON.parse(fs.readFileSync(path.join(root, 'results/real/predictions.json'), 'utf8'));
  const base = process.env.VERIFUSE_URL || 'http://127.0.0.1:8765';
  const browser = await chromium.launch({ executablePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  page.setDefaultTimeout(120000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base, { waitUntil: 'networkidle' });
    const health = await (await page.request.get(base + '/api/health')).json();
    assert.equal(health.profile, 'real');
    assert.equal(health.model_ready, true);
    await page.waitForFunction(() => document.documentElement.dataset.profile === 'real');
    const samples = page.locator('#samples button');
    assert.ok(await samples.count() > 0);
    const id = await samples.first().getAttribute('data-sample-id');
    const expected = predictions.find(row => row.id === id);
    assert.equal(expected.split, 'test');
    await samples.first().click();
    const measured = [];
    for (const model of ['feature_fusion', 'late_fusion', 'image_only', 'text_only']) {
      await page.locator('#model-select').selectOption(model);
      const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/predict'), { timeout: 120000 });
      await page.locator('#analyze-button').click();
      const response = await responsePromise;
      assert.equal(response.status(), 200);
      const result = await response.json();
      assert.equal(result.profile, 'real');
      assert.equal(result.model, model);
      assert.ok(Math.abs(result.fake_probability - expected.probabilities[model]) < 0.0001, 'Live inference matches cached evaluation');
      assert.deepEqual(result.token_attributions, []);
      assert.deepEqual(result.image_regions, []);
      await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
      assert.match(await page.locator('#result-content').innerText(), /Generalization to this input is unverified/);
      measured.push({model, probability: result.fake_probability, cached_probability: expected.probabilities[model]});
    }
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({path: path.join(output, 'prediction.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }'});
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#download-result').click();
    const download = await downloadPromise;
    const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal(exported.profile, 'real');
    assert.equal(exported.input.sample_id, id);
    await page.locator('.nav-link[data-view="experiments"]').click();
    await page.locator('#download-experiments').waitFor();
    const experimentText = await page.locator('#experiments-content').innerText();
    assert.match(experimentText, /Majority-class accuracy/i);
    assert.match(experimentText, /MACRO F1/);
    assert.match(experimentText, /External generalization remains unmeasured/);
    assert.doesNotMatch(experimentText.split('Inspect run data')[0], /undefined|NaN|When the model should abstain/);
    await page.screenshot({path: path.join(output, 'experiments.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }'});
    await page.locator('.nav-link[data-view="about"]').click();
    assert.match(await page.locator('#view-about').innerText(), /earlier synthetic benchmark/);
    await page.setViewportSize({width: 390, height: 844});
    await page.waitForFunction(() => document.querySelector('#sidebar').getBoundingClientRect().right <= 1);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true, animations: 'disabled', style: '#toast { visibility: hidden; }'});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    const missing = await page.request.post(base + '/api/predict', {data: {text: 'A caption without an image', model: 'feature_fusion'}});
    assert.equal(missing.status(), 200);
    assert.equal((await missing.json()).prediction, 'UNCERTAIN');
    assert.deepEqual(errors, []);
    const result = {passed: true, browser: 'Brave', sample: id, comparisons: measured, page_errors: errors};
    fs.writeFileSync(path.join(output, 'validation.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

"""Headless end-to-end UI test for the CatFlow web demo."""
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                  "--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 412, "height": 915})
        msgs = []
        page.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: msgs.append(f"[pageerror] {e}"))

        print("· loading page")
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector(".opt", timeout=15000)
        n_ads = page.locator("#ads-options .opt").count()
        print(f"  adsorbate options rendered: {n_ads}")

        # --- de novo generation -------------------------------------------
        print("· de novo: search '*OH' and pick")
        page.fill("#ads-search", "*OH")
        page.wait_for_timeout(300)
        page.locator("#ads-options .opt").first.click()
        assert not page.locator("#gen-btn").is_disabled(), "gen btn disabled"

        print("· clicking Generate (this runs the model)…")
        page.click("#gen-btn")
        page.wait_for_selector("#result-card:not(.hidden)", timeout=120000)
        page.wait_for_selector("#result-summary .pill", timeout=10000)
        summary = page.locator("#result-summary").text_content()
        print(f"  result summary: {summary.strip()}")
        assert "valid samples" in summary, "summary missing"

        page.wait_for_timeout(3500)  # let NGL render
        canvases = page.locator("#sample-view canvas").count()
        print(f"  NGL canvases in sample view: {canvases}")
        assert canvases >= 2, "expected 2 viewers (final + trajectory)"

        # trajectory play
        print("· trajectory: press Play")
        page.locator("#sample-view button.ghost", has_text="Play").click()
        page.wait_for_timeout(1500)

        # supercell + cell toggles
        print("· final viewer: toggle supercell + unit cell")
        page.locator("#sample-view button.ghost", has_text="supercell").click()
        page.wait_for_timeout(800)
        page.locator("#sample-view button.ghost", has_text="Unit cell").click()
        page.wait_for_timeout(500)

        # --- energy evaluation --------------------------------------------
        print("· clicking E eval (UMA relaxation)…")
        page.locator("#sample-view button.eval").click()
        page.wait_for_selector("#sample-view .energy:not(.hidden)",
                               timeout=120000)
        energy = page.locator("#sample-view .energy").text_content()
        print(f"  energy panel text: {energy.strip()}")
        assert "eV" in energy, "energy panel empty"
        has_volcano = page.locator("#sample-view svg.vplot").count() > 0
        print(f"  volcano plot present: {has_volcano}")
        assert has_volcano, "volcano expected for *OH"

        page.screenshot(path="test_screenshot.png", full_page=True)
        print("· screenshot saved -> test_screenshot.png")

        # --- structure prediction full flow -------------------------------
        print("· switching to structure-prediction mode")
        page.locator(".tab", has_text="Structure prediction").click()
        assert page.locator("#comp-card").is_visible(), "comp card hidden"
        page.fill("#comp-search", "Pt")
        page.wait_for_timeout(300)
        nc = page.locator("#comp-options .opt").count()
        print(f"  composition options for 'Pt': {nc}")
        assert 0 < nc < 220, "composition filter not applied"

        print("· picking adsorbate *CO and a Pt composition")
        page.fill("#ads-search", "*CO")
        page.wait_for_timeout(300)
        page.locator("#ads-options .opt").first.click()
        page.locator("#comp-options .opt").first.click()
        assert not page.locator("#gen-btn").is_disabled(), "gen btn disabled"

        print("· clicking Generate (structure prediction)…")
        page.click("#gen-btn")
        page.wait_for_selector("#result-card:not(.hidden)", timeout=120000)
        page.wait_for_selector("#result-summary .pill", timeout=10000)
        sp_summary = page.locator("#result-summary").text_content()
        print(f"  SP result summary: {sp_summary.strip()}")
        assert "Slab" in sp_summary, "SP summary missing composition"
        page.wait_for_timeout(2500)
        sp_canvases = page.locator("#sample-view canvas").count()
        print(f"  SP NGL canvases: {sp_canvases}")
        assert sp_canvases >= 2, "SP viewers missing"

        print("· SP: clicking E eval…")
        page.locator("#sample-view button.eval").click()
        page.wait_for_selector("#sample-view .energy:not(.hidden)",
                               timeout=120000)
        sp_energy = page.locator("#sample-view .energy").text_content()
        print(f"  SP energy: {sp_energy.strip()[:120]}…")
        assert "eV" in sp_energy, "SP energy empty"

        # --- custom (free-input) composition ------------------------------
        print("· custom composition: switch to 'Custom formula'")
        page.locator("#comp-source .subtab", has_text="Custom").click()
        assert page.locator("#comp-custom-panel").is_visible(), \
            "custom panel hidden"
        page.fill("#comp-custom", "Pd2Ag2")
        page.wait_for_timeout(300)
        preview = page.locator("#comp-custom-preview").text_content()
        print(f"  custom preview: {preview.strip()}")
        assert "atom" in preview, "custom preview not shown"
        assert not page.locator("#gen-btn").is_disabled(), \
            "gen btn disabled for valid custom composition"

        print("· clicking Generate (custom composition Pd2Ag2)…")
        page.click("#gen-btn")
        page.wait_for_selector("#result-card:not(.hidden)", timeout=120000)
        page.wait_for_selector("#result-summary .pill", timeout=10000)
        cu_summary = page.locator("#result-summary").text_content()
        print(f"  custom-comp result: {cu_summary.strip()}")
        assert "valid samples" in cu_summary, "custom-comp generation failed"

        # invalid custom formula should disable the button
        page.fill("#comp-custom", "Xx9")
        page.wait_for_timeout(300)
        disabled = page.locator("#gen-btn").is_disabled()
        print(f"  invalid formula 'Xx9' disables Generate: {disabled}")
        assert disabled, "invalid composition should disable Generate"

        errors = [m for m in msgs if "pageerror" in m or "[error]" in m]
        if errors:
            print("\n!! console errors:")
            for e in errors:
                print("   " + e)
        else:
            print("\n✓ no console / page errors")

        browser.close()
        return len(errors) == 0


if __name__ == "__main__":
    ok = run()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

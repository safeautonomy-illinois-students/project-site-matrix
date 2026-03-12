# ECE 484 Project Website Template

A clean, responsive Quarto template for academic project pages. Inspired by the popular [Nerfies](https://nerfies.github.io/) project page.

## Getting Started

### Step 1: Enable GitHub Pages on your repo

Your site won't go live until you flip this switch:

1. Go to your repo on GitHub
2. Click **Settings** (top menu bar)
3. In the left sidebar, click **Pages**
4. Under **Source**, select **GitHub Actions** from the dropdown
5. Click **Save**

That's it, from now on, every push to `main` will automatically build and publish your site. Your site URL will be:

```
https://<org-name>.github.io/<repo-name>/
```

> **Note:** The first deploy takes 1–2 minutes. If you see a 404, check the **Actions** tab in your repo to see if the build is still running.

### Step 2: Clone the repo and start editing

```bash
git clone https://github.com/<org-name>/<repo-name>.git
cd <repo-name>
```

Open `**index.qmd**` in any text editor and start replacing the placeholder content. Every section is marked with comments like `<!-- ABSTRACT -->` so you know exactly what to change.

At a minimum, update these:

1. **Title and subtitle** at the top of `index.qmd`
2. **Author names and affiliations**
3. **Abstract text**
4. **Teaser image** — replace `static/images/placeholder-teaser.svg` with your own figure
5. **Results images** in `static/images/`
6. **Action buttons** — update the links to your paper, code, video, etc. (remove any that don't apply)
7. **BibTeX entry** at the bottom

### Step 3: Preview locally (optional but recommended)

If you want to preview before pushing:

1. Install Quarto from [quarto.org/docs/get-started](https://quarto.org/docs/get-started/)
2. Install the Font Awesome extension (for button icons):
  ```bash
   quarto add quarto-ext/fontawesome
  ```
3. Run:
  ```bash
   quarto preview
  ```
   This opens a live-reloading preview in your browser.

### Step 4: Push your changes

```bash
git add -A
git commit -m "Update project page"
git push
```

GitHub Actions will automatically rebuild and deploy your site within a couple of minutes.

---

## File Structure

```
├── _quarto.yml                    # Quarto project config
├── index.qmd                      # Main page content (edit this!)
├── styles.css                     # Custom CSS theme
├── static/
│   ├── images/                    # Your figures and result images
│   └── videos/                    # Your video files (optional)
├── .github/workflows/deploy.yml   # Auto-deploy workflow (don't touch)
└── README.md                      # This file
```

## Customization Tips

### Changing colors

Edit the CSS variables at the top of `styles.css`:

```css
:root {
  --color-link: #4a6cf7;       /* Accent / link color */
  --color-btn-bg: #4a6cf7;     /* Button background */
  --color-heading: #363636;    /* Heading color */
  --max-width: 960px;          /* Content max-width */
}
```

### Adding more result images

Drop images into `static/images/`, then add cards inside the `.results-grid` div in `index.qmd`:

```markdown
::: {.result-card}
![Caption text](static/images/my-result.png)
:::
```

### Embedding videos

Use Quarto's built-in video shortcode:

```markdown
{{< video https://www.youtube.com/embed/VIDEO_ID >}}
{{< video static/videos/demo.mp4 >}}
```

### Removing sections you don't need

Each section is wrapped in a `::: {.section-container}` block. Delete the entire block (from the opening `:::` to the closing `:::`) to remove it.

### Removing button icons

If the Font Awesome icons aren't rendering, remove the `{{< fa ... >}}` shortcodes from the buttons in `index.qmd` — the buttons will still work fine as plain text.

## Troubleshooting


| Problem                | Fix                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Site shows 404         | Check **Settings → Pages** is set to "GitHub Actions". Then check the **Actions** tab to see if the build succeeded. |
| Build fails in Actions | Click into the failed run to read the error log. Common fix: make sure `index.qmd` has no syntax errors.             |
| Images not showing     | Check that the file paths in `index.qmd` match the actual filenames in `static/images/`. Paths are case-sensitive.   |
| Changes not appearing  | Make sure you pushed to the `main` branch. The workflow only triggers on `main`.                                     |


## Credits

This template is inspired by the [Nerfies project page](https://nerfies.github.io/), which is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
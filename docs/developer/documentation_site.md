# Documentation Site

We use [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) to generate this documentation site.

The site is automatically deployed to [GitHub Pages](https://pages.github.com/) when a new commit contains changes to the `docs/` directory. See the [corresponding GitHub Actions](https://github.com/jerry871002/bayesian-strategy-inference/blob/master/.github/workflows/mkdoc_ghpages.yaml) for more details.

## Run Locally

To run the site locally, first install the package.

```bash
pip install mkdocs-material
```

Then run the following command and open `http://127.0.0.1:8000/` in your browser.

```bash
mkdocs serve
```

# Math Blog

A personal math and CS notes blog built with [Astro](https://astro.build). Supports LaTeX equations via KaTeX and Obsidian-flavored markdown syntax.

**Live site:** [chenxr.cloud](https://chenxr.cloud)

## Features

- Static site generation with Astro
- LaTeX math rendering (KaTeX, bundled locally)
- Obsidian `![[image.png]]` and `[[wikilink]]` syntax support
- Topic-based organization with breadcrumb navigation
- Poole-inspired clean typography with dark mode
- LXGW WenKai Chinese font support

## Tech Stack

- **Framework:** Astro 6
- **Math:** remark-math + rehype-katex (KaTeX)
- **Styling:** Single global CSS file with CSS custom properties
- **Deployment:** Cloudflare Pages

## Project Structure

```
src/
  content/blog/       # Markdown notes organized by topic
  components/         # Shared Astro layout and category tree components
  pages/
    index.astro       # Blog homepage
    [...id].astro     # Individual note pages
    archives.astro    # Archive/category tree
    tags.astro        # Tag index
    calendar.astro    # Timeline and activity calendar
    about.astro       # About page
  styles/global.css   # Global stylesheet
public/
  fonts/              # Custom fonts (KaTeX, LXGW WenKai)
  images/             # Blog images
  katex.min.css       # Bundled KaTeX styles
```

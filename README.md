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
  pages/
    index.astro       # Landing page
    xK9m2p/
      index.astro     # Blog homepage with topic grid
      [...id].astro   # Individual note pages
      topics/[topic].astro  # Topic listing pages
  styles/global.css   # Global stylesheet
public/
  fonts/              # Custom fonts (KaTeX, LXGW WenKai)
  images/             # Blog images
  katex.min.css       # Bundled KaTeX styles
```

## Writing Notes

Create a markdown file in `src/content/blog/<topic>/` with this frontmatter:

```markdown
---
title: "Note Title"
description: "Short description"
pubDate: 2026-04-01
topic: "Topic Name"
tags: ["tag1", "tag2"]
---

# First Heading

Your content here...
```

### Supported Syntax

- **LaTeX:** `$inline$` and `$$display$$` math
- **Obsidian images:** `![[image.png]]` and `![[image.png|300]]`
- **Wikilinks:** `[[Page Name]]` (rendered as plain text)

## Commands

```bash
npm run dev      # Start dev server
npm run build    # Build static site to dist/
npm run preview  # Preview built site locally
```

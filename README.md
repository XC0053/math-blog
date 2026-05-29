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
    categories.astro  # Category tree
    tags.astro        # Tag index
    writing.astro     # Archive timeline
    about.astro       # About page
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

## How to Publish a New Note (No AI Needed)

Open a terminal and run these steps:

### 1. Go to the project folder

```bash
cd E:/vibecoding
```

### 2. Create a new note

Notes live inside `src/content/blog/<topic>/`. The folder name is the topic.

**Option A: Add to an existing topic** (e.g. "Algorithm Notes")

```bash
# Create the markdown file
notepad src/content/blog/algorithm-notes/my-new-note/my-new-note.md
```

**Option B: Create a brand new topic**

```bash
# Make a new topic folder first
mkdir -p src/content/blog/new-topic-name
notepad src/content/blog/new-topic-name/my-note/my-note.md
```

### 3. Write the note

Start the file with this frontmatter (the `---` block is required):

```markdown
---
title: "My Note Title"
description: "One-line description"
pubDate: 2026-04-02
topic: "Topic Name"
tags: ["tag1", "tag2"]
---

# First Heading

Your content here. Use $inline math$ or $$display math$$.
```

### 4. Add images (if any)

Place images in the matching topic folder:

```bash
# Copy images to the right spot
cp /path/to/image.png src/content/blog/algorithm-notes/my-new-note/

# Then reference in markdown with Obsidian syntax:
# ![[image.png]] or ![[image.png|300]]
```

### 5. Preview locally

```bash
npm run dev
# Open http://localhost:4321/ in your browser
# Check that everything looks right, then Ctrl+C to stop
```

### 6. Build and push online

```bash
npm run build
git add .
git commit -m "Add note about X"
git push
```

If `git push` fails with a proxy error, run:

```bash
git -c http.proxy="" -c https.proxy="" push
```

The site at [chenxr.cloud](https://chenxr.cloud) will update automatically after the push.

### Quick reference: editing existing notes

```bash
# Edit an existing note
notepad src/content/blog/algorithm-notes/transformer/transformer.md

# Edit the CSS theme
notepad src/styles/global.css

# Then build and push
npm run build && git add . && git commit -m "Update X" && git push
```

import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import cloudflare from "@astrojs/cloudflare";

function slugifyPath(path) {
  return path
    .split(/[\\/]/)
    .filter(Boolean)
    .map((segment) =>
      segment
        .toLowerCase()
        .trim()
        .replace(/\s+/g, "-")
        .replace(/[^a-z0-9-]/g, "")
    )
    .join("/");
}

function imagePath(base, filename) {
  return [base, filename]
    .filter(Boolean)
    .join("/")
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

// Convert Obsidian ![[image.png]] and [[wikilink]] before markdown rendering.
function remarkObsidianSyntax() {
  return function (tree, file) {
    const srcPath = file.history && file.history[0] ? file.history[0] : "";
    const blogMatch = srcPath.match(/[/\\]blog[/\\](.+)[/\\]/);
    const imgBase = blogMatch ? "/images/" + slugifyPath(blogMatch[1]) : "";

    const visit = function (node) {
      if (node.children) {
        const newChildren = [];
        for (const child of node.children) {
          if (child.type === "text") {
            const regex = /!\[\[([^\]|]+)(?:\|(\d+))?\]\]/g;
            let lastIndex = 0;
            let match;
            let found = false;
            while ((match = regex.exec(child.value)) !== null) {
              found = true;
              if (match.index > lastIndex) {
                newChildren.push({ type: "text", value: child.value.slice(lastIndex, match.index) });
              }
              const imgSrc = imagePath(imgBase, match[1]);
              newChildren.push({
                type: "image",
                url: imgSrc,
                alt: "",
                data: {
                  hProperties: {
                    style: "max-width:100%;height:auto;display:block;margin:1.5rem 0",
                    ...(match[2] ? { width: match[2] } : {}),
                  },
                },
              });
              lastIndex = regex.lastIndex;
            }
            if (found) {
              if (lastIndex < child.value.length) {
                newChildren.push({ type: "text", value: child.value.slice(lastIndex) });
              }
              continue;
            }
            if (/\[\[([^\]]+)\]\]/.test(child.value)) {
              child.value = child.value.replace(/\[\[([^\]]+)\]\]/g, "$1");
            }
          }
          visit(child);
          newChildren.push(child);
        }
        node.children = newChildren;
      }
    };
    visit(tree);
  };
}

const isDev = process.env.npm_lifecycle_event === "dev";

export default defineConfig({
  site: "https://chenxr.cloud",

  markdown: {
    remarkPlugins: [remarkObsidianSyntax, remarkMath],
    rehypePlugins: [rehypeKatex],
  },

  adapter: isDev ? undefined : cloudflare(),
});

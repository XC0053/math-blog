import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import cloudflare from "@astrojs/cloudflare";

// Convert Obsidian ![[image.png]] and ![[image.png|width]] in raw text to <img>
// Must run as a rehype plugin to catch cases where remark already parsed ![[ into broken nodes
function rehypeObsidianImages() {
  return function (tree, file) {
    const srcPath = file.history && file.history[0] ? file.history[0] : "";
    const blogMatch = srcPath.match(/[/\\]blog[/\\](.+)[/\\]/);
    const imgBase = blogMatch ? "/images/" + blogMatch[1].replace(/\\/g, "/") : "";

    const visit = function (node) {
      if (node.children) {
        const newChildren = [];
        for (const child of node.children) {
          // Handle text nodes containing Obsidian image syntax
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
              const imgSrc = imgBase ? imgBase + "/" + match[1] : match[1];
              const widthAttr = match[2] ? ' width="' + match[2] + '"' : "";
              newChildren.push({
                type: "element",
                tagName: "img",
                properties: {
                  src: imgSrc,
                  alt: "",
                  style: "max-width:100%;height:auto;display:block;margin:1.5rem 0",
                  ...(match[2] ? { width: match[2] } : {}),
                },
                children: [],
              });
              lastIndex = regex.lastIndex;
            }
            if (found) {
              if (lastIndex < child.value.length) {
                newChildren.push({ type: "text", value: child.value.slice(lastIndex) });
              }
              continue;
            }
            // Convert [[wikilink]] to plain text
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

export default defineConfig({
  site: "https://chenxr.cloud",

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeObsidianImages],
  },

  adapter: cloudflare()
});
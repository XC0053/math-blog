import { defineConfig } from "astro/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// Convert Obsidian ![[image.png]] and ![[image.png|width]] to standard markdown
function remarkObsidianImages() {
  return function (tree) {
    const visit = function (node) {
      if (node.children) {
        const newChildren = [];
        for (const child of node.children) {
          if (child.type === "text") {
            const regex = /!\[\[([^\]|]+)(?:\|(\d+))?\]\]/g;
            let lastIndex = 0;
            let match;
            const parts = [];
            while ((match = regex.exec(child.value)) !== null) {
              if (match.index > lastIndex) {
                parts.push({ type: "text", value: child.value.slice(lastIndex, match.index) });
              }
              if (match[2]) {
                parts.push({
                  type: "html",
                  value: '<img src="' + match[1] + '" alt="" width="' + match[2] + '" style="max-width:100%;height:auto;display:block;margin:1.5rem 0" />',
                });
              } else {
                parts.push({ type: "image", url: match[1], alt: "" });
              }
              lastIndex = regex.lastIndex;
            }
            if (lastIndex > 0) {
              if (lastIndex < child.value.length) {
                parts.push({ type: "text", value: child.value.slice(lastIndex) });
              }
              for (const p of parts) newChildren.push(p);
              continue;
            }
          }
          // Convert [[wikilink]] to plain text
          if (child.type === "text" && /\[\[([^\]]+)\]\]/.test(child.value)) {
            child.value = child.value.replace(/\[\[([^\]]+)\]\]/g, "$1");
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
    remarkPlugins: [remarkMath, remarkObsidianImages],
    rehypePlugins: [rehypeKatex],
  },
});

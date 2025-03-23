"""Test Mistune Markdown rendering for HAL's prose extraction.

Loads a cached Markdown file (e.g., from build_db.py), processes it with a custom Mistune
renderer to strip code blocks and tables, and saves the prose-only output to a test file.
Used to debug or compare Markdown parsing for HAL's chunking pipeline, isolating text content.
"""
import os
import mistune
import mistune.renderers.markdown
import mistune.renderers

# Pick a cached .md file—swap this path to yours!
input_file = os.path.expanduser("/home/totally/code/hal/cache/Node.js v22.14.0 Documentation.md")  # or Elements.md, whatever's juicy
output_dir = "test_outputs"
os.makedirs(output_dir, exist_ok=True)

# Read the cached .md file
with open(input_file, "r", encoding="utf-8") as f:
    md_text = f.read()

# 1. Mistune: Custom renderer, blank code/tables
class ProseRenderer(mistune.renderers.markdown.MarkdownRenderer):
    def block_code(self, code, lang=None):
        return ""  # Skip code blocks
    def table(self, header, body):
        return ""  # Skip tables
    def paragraph(self, token, state):
        return self.render_children(token, state) + "\n\n"  # Keep prose

mistune_renderer = ProseRenderer()
mistune_md = mistune.Markdown(renderer=mistune_renderer)
mistune_output = mistune_md(md_text).strip()
with open(os.path.join(output_dir, "output_mistune.md"), "w", encoding="utf-8") as f:
    f.write(mistune_output)
print("Mistune done!")


print(f"Outputs saved in {output_dir}. Diff away!")
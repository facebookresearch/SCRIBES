# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import re
from collections import defaultdict

from bs4 import BeautifulSoup, Comment


class HTMLCompressor:
    def __init__(self):
        # Attributes to remove (presentational/behavioral)
        self.remove_attrs = {
            "style",
            "onclick",
            "onload",
            "onmouseover",
            "onmouseout",
            "onfocus",
            "onblur",
            "onchange",
            "onsubmit",
            "width",
            "height",
            "bgcolor",
            "color",
            "font",
            "align",
            "valign",
            "border",
            "cellpadding",
            "cellspacing",
            "marginwidth",
            "marginheight",
        }

        # Attributes to keep (structural/semantic)
        self.keep_attrs = {
            "id",
            "class",
            "data-*",
            "role",
            "aria-*",
            "name",
            "type",
            "href",
            "src",
            "alt",
            "title",
            "rel",
            "target",
            "for",
            "action",
            "method",
            "value",
            "placeholder",
            "required",
        }

        # Tags to completely remove
        self.remove_tags = {
            "script",
            "style",
            "noscript",
            "iframe",
            "embed",
            "object",
            "applet",
            "meta",
            "link",
            "base",
        }

        # Tags that typically contain repetitive content
        self.repetitive_containers = {
            "ul",
            "ol",
            "div",
            "section",
            "tbody",
            "thead",
            "select",
        }

    def should_keep_attribute(self, attr_name):
        """Determine if an attribute should be kept"""
        if attr_name in self.remove_attrs:
            return False
        if attr_name in self.keep_attrs:
            return True
        # Keep data-* and aria-* attributes
        if attr_name.startswith("data-") or attr_name.startswith("aria-"):
            return True
        return False

    def clean_attributes(self, tag):
        """Remove unnecessary attributes from a tag"""
        if not hasattr(tag, "attrs") or not tag.attrs:
            return

        attrs_to_remove = []
        for attr_name in tag.attrs:
            if not self.should_keep_attribute(attr_name):
                attrs_to_remove.append(attr_name)

        for attr in attrs_to_remove:
            del tag.attrs[attr]

    def compress_repetitive_content(self, container, max_items=3):
        """Compress repetitive content while preserving structure"""
        try:
            if (
                not hasattr(container, "name")
                or not container.name
                or container.name not in self.repetitive_containers
            ):
                return

            # Get direct children that are tags (not text)
            children = []
            for child in container.children:
                if (
                    hasattr(child, "name")
                    and hasattr(child, "get")
                    and child.name is not None
                ):
                    children.append(child)

            if len(children) <= max_items:
                return

            # Group children by tag name and class
            groups = defaultdict(list)
            for child in children:
                try:
                    # Safely get class attribute
                    class_attr = child.get("class", [])
                    if class_attr is None:
                        class_attr = []
                    elif isinstance(class_attr, str):
                        class_attr = [class_attr]
                    elif not isinstance(class_attr, list):
                        class_attr = []

                    key = (child.name, tuple(sorted(class_attr)))
                    groups[key].append(child)
                except (AttributeError, TypeError):
                    # Skip problematic children
                    continue

            # Process each group
            for key, items in groups.items():
                if len(items) > max_items:
                    try:
                        # Keep first few items
                        keep_items = items[:max_items]
                        remove_items = items[max_items:]

                        # Remove excess items
                        for item in remove_items:
                            try:
                                item.extract()
                            except (AttributeError, TypeError):
                                continue

                        # Add comment indicating compression
                        if keep_items and keep_items[-1].parent:
                            tag_name = key[0] if key[0] else "unknown"
                            classes = " ".join(key[1]) if key[1] else ""
                            class_attr = f' class="{classes}"' if classes else ""
                            comment_text = f" ... {len(remove_items)} more <{tag_name}{class_attr}> elements ... "
                            comment = Comment(comment_text)
                            try:
                                keep_items[-1].insert_after(comment)
                            except (AttributeError, TypeError):
                                continue
                    except (IndexError, AttributeError, TypeError):
                        continue
        except Exception:
            # If anything goes wrong, just skip compression for this container
            return

    def remove_empty_lines(self, html_string):
        """Remove excessive whitespace and empty lines"""
        # Remove multiple consecutive newlines
        html_string = re.sub(r"\n\s*\n\s*\n", "\n\n", html_string)

        # Remove leading/trailing whitespace on lines
        lines = html_string.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped or (cleaned_lines and cleaned_lines[-1].strip()):
                cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def compress_html(self, html_content, max_repetitive_items=3):
        """Main compression function"""
        try:
            # Handle None or empty input
            if not html_content:
                return ""

            # Ensure input is string
            if not isinstance(html_content, str):
                html_content = str(html_content)

            # Parse HTML with error handling
            try:
                soup = BeautifulSoup(html_content, "html.parser")
            except Exception:
                # If parsing fails, try with lxml parser as fallback
                try:
                    soup = BeautifulSoup(html_content, "lxml")
                except Exception:
                    # If all parsing fails, return original content
                    return html_content

            # Remove unwanted tags completely
            for tag_name in self.remove_tags:
                try:
                    for tag in soup.find_all(tag_name):
                        if tag:
                            tag.decompose()
                except Exception:
                    continue

            # Remove comments (except our own compression comments)
            try:
                comments_to_remove = []
                for comment in soup.find_all(
                    string=lambda text: isinstance(text, Comment)
                ):
                    if comment and not str(comment).strip().startswith("..."):
                        comments_to_remove.append(comment)

                for comment in comments_to_remove:
                    try:
                        comment.extract()
                    except Exception:
                        continue
            except Exception:
                pass

            # Process all tags
            try:
                all_tags = soup.find_all()
                for tag in all_tags:
                    if not tag:
                        continue

                    try:
                        # Clean attributes
                        self.clean_attributes(tag)
                    except Exception:
                        continue

                    try:
                        # Compress repetitive content
                        self.compress_repetitive_content(tag, max_repetitive_items)
                    except Exception:
                        continue
            except Exception:
                pass

            # Convert back to string
            try:
                html_string = str(soup)
            except Exception:
                # If conversion fails, return original content
                return html_content

            # Clean up whitespace
            try:
                html_string = self.remove_empty_lines(html_string)
            except Exception:
                pass

            return html_string

        except Exception:
            # If anything goes catastrophically wrong, return original content
            return html_content if html_content else ""

    def compress_file(self, input_file, output_file=None, max_repetitive_items=3):
        """Compress HTML from file"""
        with open(input_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        compressed_html = self.compress_html(html_content, max_repetitive_items)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(compressed_html)
            print(f"Compressed HTML saved to {output_file}")
        else:
            print(compressed_html)

        # Print compression stats
        original_size = len(html_content)
        compressed_size = len(compressed_html)
        compression_ratio = (1 - compressed_size / original_size) * 100

        print(f"\nCompression Stats:")
        print(f"Original size: {original_size:,} characters")
        print(f"Compressed size: {compressed_size:,} characters")
        print(f"Compression ratio: {compression_ratio:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compress HTML while preserving structure for LLM script generation"
    )
    parser.add_argument("input_file", help="Input HTML file")
    parser.add_argument("-o", "--output", help="Output file (default: print to stdout)")
    parser.add_argument(
        "-m",
        "--max-items",
        type=int,
        default=3,
        help="Maximum number of repetitive items to keep (default: 3)",
    )

    args = parser.parse_args()

    compressor = HTMLCompressor()
    compressor.compress_file(args.input_file, args.output, args.max_items)


if __name__ == "__main__":
    # Example usage when run as script
    compressor = HTMLCompressor()

    # Example HTML content
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Page</title>
        <style>body { font-family: Arial; }</style>
        <script>console.log('hello');</script>
    </head>
    <body style="background-color: white;" onclick="track()">
        <div class="header" id="main-header">
            <h1>Products</h1>
        </div>
        <div class="product-grid" data-category="electronics">
            <div class="product-card" data-id="1" style="border: 1px solid #ccc;">
                <h3 class="product-title">Product 1</h3>
                <span class="price">$19.99</span>
            </div>
            <div class="product-card" data-id="2" style="border: 1px solid #ccc;">
                <h3 class="product-title">Product 2</h3>
                <span class="price">$24.99</span>
            </div>
            <div class="product-card" data-id="3" style="border: 1px solid #ccc;">
                <h3 class="product-title">Product 3</h3>
                <span class="price">$29.99</span>
            </div>
            <div class="product-card" data-id="4" style="border: 1px solid #ccc;">
                <h3 class="product-title">Product 4</h3>
                <span class="price">$34.99</span>
            </div>
            <div class="product-card" data-id="5" style="border: 1px solid #ccc;">
                <h3 class="product-title">Product 5</h3>
                <span class="price">$39.99</span>
            </div>
        </div>
    </body>
    </html>
    """

    print("Original HTML:")
    print(sample_html)
    print("\n" + "=" * 50 + "\n")

    print("Compressed HTML:")
    compressed = compressor.compress_html(sample_html)
    print(compressed)

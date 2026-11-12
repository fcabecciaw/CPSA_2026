/**
 * ARI's current sentence, shown as subtitles under the sphere.
 *
 * Its own component and its own DOM subtree (`#subtitles`), a sibling of the
 * avatar and of the objects panel: the objects panel refreshes twice a second
 * and must never rewrite, move or resize the line people are reading. The
 * node is written only when the spoken line changes; clearing fades it out
 * and keeps the old words until the next line, so the fade has text to fade.
 */

export class Subtitles {
  private readonly node: HTMLElement;
  private line: string | null = null;

  constructor(node: HTMLElement) {
    this.node = node;
  }

  /** The line on screen, or null while hidden. */
  get text(): string | null {
    return this.line;
  }

  show(text: string): void {
    if (text === "") {
      this.clear();
      return;
    }
    if (text === this.line) {
      return;
    }
    this.line = text;
    // The box has a fixed height: longer lines step down in size instead of
    // growing it (the longest fact is ~150 characters).
    this.node.dataset.length = text.length > 80 ? "long" : text.length > 45 ? "medium" : "short";
    if (this.node.textContent !== text) {
      this.node.textContent = text;
    }
    this.node.classList.remove("bubble-empty");
  }

  clear(): void {
    if (this.line === null) {
      return;
    }
    this.line = null;
    this.node.classList.add("bubble-empty");
  }
}

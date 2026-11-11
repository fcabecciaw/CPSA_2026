/** Small DOM helpers. Text always goes through `textContent` (no innerHTML). */

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className = "",
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className !== "") {
    node.className = className;
  }
  if (text !== "") {
    node.textContent = text;
  }
  return node;
}

export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node === null) {
    throw new Error(`Elemento #${id} mancante in index.html`);
  }
  return node as T;
}

export function clear(node: Element): void {
  node.replaceChildren();
}

export function chip(text: string, className = "chip"): HTMLSpanElement {
  return el("span", className, text);
}

export function code(text: string): HTMLElement {
  return el("code", "code", text);
}

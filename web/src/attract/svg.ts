/** A tiny SVG builder: diagrams are made of elements, never innerHTML. */

const NS = "http://www.w3.org/2000/svg";

type Attrs = Record<string, string | number>;
type Child = SVGElement | string;

export function svg(tag: string, attrs: Attrs = {}, ...children: Child[]): SVGElement {
  const node = document.createElementNS(NS, tag);
  for (const [name, value] of Object.entries(attrs)) {
    node.setAttribute(name, String(value));
  }
  for (const child of children) {
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

/** The root of one diagram: 640×480 units, scaled to its box. */
export function canvas(label: string, ...children: SVGElement[]): SVGSVGElement {
  const root = svg(
    "svg",
    {
      viewBox: "0 0 640 480",
      class: "diagram",
      role: "img",
      "aria-label": label,
    },
    ...children,
  ) as SVGSVGElement;
  return root;
}

/** A grid of lines every `step` units over a rectangle. */
export function grid(x: number, y: number, width: number, height: number, step: number): SVGElement {
  const parts: string[] = [];
  for (let gx = x + step; gx < x + width - 0.5; gx += step) {
    parts.push(`M${gx} ${y}V${y + height}`);
  }
  for (let gy = y + step; gy < y + height - 0.5; gy += step) {
    parts.push(`M${x} ${gy}H${x + width}`);
  }
  return svg("path", { d: parts.join(""), class: "d-grid" });
}

/** A text label in the page's font. */
export function label(
  x: number,
  y: number,
  text: string,
  className = "d-label",
  anchor: "start" | "middle" | "end" = "middle",
): SVGElement {
  return svg("text", { x, y, class: className, "text-anchor": anchor }, text);
}

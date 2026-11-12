/**
 * "Come funziona": the scenes of the attract loop, for high-school students.
 *
 * Every number is either live (the board's WebSocket) or measured on the
 * board and in the code: 640×480 at 60 fps from the RealSense RGB node, the
 * YOLOX input of 416×416 letterboxed with grey (pad 114) and quantised as
 * `round(pixel ÷ 4)` (`src/cpsa/yolox.py`), its three output grids 52, 26 and
 * 13 cells wide, 80 COCO classes, the 17 MoveNet keypoints, ~4.7 ms of DPU per
 * image, and the KV260 DPU (B4096 at 300 MHz) peak of about 1.2 TOPS int8.
 */

import { formatNumber } from "../labels";
import { canvas, grid, label, svg } from "./svg";

/** What the board says right now; null when not connected or not known. */
export interface LiveNumbers {
  fps: number | null;
  dpuMs: number | null;
  /** The kinds of object in the frame now, most frequent first. */
  labels: string[];
}

export interface SceneSpec {
  id: string;
  headline: string;
  lines: readonly string[];
  durationMs: number;
  diagram: () => SVGSVGElement;
  /** A line built from live numbers, refreshed while the scene is shown. */
  live?: (live: LiveNumbers) => string | null;
}

/** Measured on the board (`scripts/profile_showcase.py`), used when offline. */
export const MEASURED = { fps: 60, dpuMs: 4.7 } as const;

const SCENE_MS = 10_000;

function fps(live: LiveNumbers): string {
  return formatNumber(live.fps ?? MEASURED.fps, 0);
}

function dpu(live: LiveNumbers): string {
  return formatNumber(live.dpuMs ?? MEASURED.dpuMs, 1);
}

export const SCENES: readonly SceneSpec[] = [
  {
    id: "camera",
    headline: "La telecamera scatta 60 foto al secondo",
    lines: [
      "Ogni foto è una griglia di 640 × 480 pixel: sono circa 18 milioni di pixel al secondo.",
      "Ogni pixel è fatto di tre numeri, per il rosso, il verde e il blu.",
    ],
    durationMs: SCENE_MS,
    diagram: cameraDiagram,
    live: (live) =>
      live.fps === null
        ? "Misurato sulla scheda: 60 immagini analizzate al secondo"
        : `Adesso: ${fps(live)} immagini analizzate al secondo`,
  },
  {
    id: "prepare",
    headline: "Prima di tutto, l'immagine si semplifica",
    lines: [
      "Viene ridotta a 416 × 416 pixel, con due bande grigie per farla quadrata.",
      "Ogni valore diventa un intero a 8 bit (int8): numeri piccoli, calcoli molto più veloci.",
    ],
    durationMs: SCENE_MS,
    diagram: prepareDiagram,
  },
  {
    id: "chip",
    headline: "Un chip che fa migliaia di moltiplicazioni insieme",
    lines: [
      "La scheda AMD Kria KV260 ha un FPGA, un chip i cui circuiti si possono riconfigurare.",
      "Dentro c'è la DPU, un acceleratore per reti neurali: fino a circa 1,2 mila miliardi di operazioni al secondo.",
    ],
    durationMs: SCENE_MS,
    diagram: chipDiagram,
    live: (live) =>
      live.dpuMs === null && live.fps === null
        ? `Misurato: un'immagine in circa ${dpu(live)} ms, ${fps(live)} al secondo`
        : `Adesso: un'immagine in ${dpu(live)} ms, ${fps(live)} al secondo`,
  },
  {
    id: "network",
    headline: "Una rete neurale guarda strato dopo strato",
    lines: [
      "I primi strati vedono i bordi, poi le forme, infine gli oggetti interi.",
      "YOLOX-nano conosce 80 categorie e per ogni oggetto dà un riquadro e quanto ne è sicura. MoveNet trova 17 punti del corpo.",
    ],
    durationMs: SCENE_MS,
    diagram: networkDiagram,
    live: (live) =>
      live.labels.length === 0 ? null : `Adesso vede: ${live.labels.slice(0, 4).join(", ")}`,
  },
  {
    id: "tracking",
    headline: "Da riquadri tremolanti a oggetti stabili",
    lines: [
      "Da un fotogramma all'altro i riquadri tremano un po'.",
      "Il tracciamento capisce che è lo stesso oggetto: il riquadro resta fermo finché l'oggetto non si muove davvero.",
    ],
    durationMs: SCENE_MS,
    diagram: trackingDiagram,
  },
  {
    id: "mind",
    headline: "Come ARI decide cosa dire",
    lines: [
      "Il portatile riceve dalla scheda oggetti e pose, li tiene in una memoria breve e con delle regole sceglie la frase giusta.",
      "Poi ARI la pronuncia con una voce sintetizzata sul portatile stesso.",
    ],
    durationMs: SCENE_MS,
    diagram: mindDiagram,
  },
  {
    id: "privacy",
    headline: "Niente viene registrato",
    lines: [
      "Ogni immagine viene analizzata e subito dimenticata.",
      "Nessuna foto e nessun video vengono salvati, niente viene inviato su internet.",
    ],
    durationMs: SCENE_MS,
    diagram: privacyDiagram,
  },
];

/* --- the diagrams --------------------------------------------------------- */

function spectrum(id: string): SVGElement {
  return svg(
    "linearGradient",
    { id, x1: 0, y1: 0, x2: 1, y2: 1 },
    svg("stop", { offset: "0", class: "stop-ion" }),
    svg("stop", { offset: "0.52", class: "stop-violet" }),
    svg("stop", { offset: "1", class: "stop-plasma" }),
  );
}

function cameraDiagram(): SVGSVGElement {
  const frame = { x: 40, y: 70, w: 400, h: 300 };
  const pixels: SVGElement[] = [];
  // The centre pixel is the dashboard's cyan, #4fd6ff: exactly 79, 214, 255.
  const colours = ["d-px-violet", "d-px-plasma", "d-px-ion"];
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      pixels.push(
        svg("rect", {
          x: 476 + col * 40,
          y: 92 + row * 40,
          width: 36,
          height: 36,
          rx: 4,
          class: `d-px ${colours[(row + col) % 3]}${row === 1 && col === 1 ? " d-px-pick" : ""}`,
        }),
      );
    }
  }
  return canvas(
    "Una foto 640 per 480 pixel e tre pixel ingranditi",
    svg("defs", {}, spectrum("att-cam")),
    svg("rect", { x: frame.x - 28, y: frame.y - 28, width: frame.w, height: frame.h, rx: 12, class: "d-ghost d-ghost-2" }),
    svg("rect", { x: frame.x - 14, y: frame.y - 14, width: frame.w, height: frame.h, rx: 12, class: "d-ghost d-ghost-1" }),
    svg("rect", { x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 12, fill: "url(#att-cam)", class: "d-image" }),
    grid(frame.x, frame.y, frame.w, frame.h, 20),
    svg("rect", { x: frame.x, y: frame.y, width: frame.w, height: frame.h, rx: 12, class: "d-outline" }),
    svg("rect", { x: frame.x, y: frame.y, width: frame.w, height: 4, class: "d-scan" }),
    svg("path", { d: "M330 140 L470 112 M330 160 L470 206", class: "d-lead" }),
    svg("rect", { x: 330, y: 140, width: 20, height: 20, class: "d-pick" }),
    ...pixels,
    label(536, 248, "79   214   255", "d-label d-mono"),
    label(536, 272, "rosso  verde  blu", "d-label d-small"),
    label(frame.x + frame.w / 2, frame.y + frame.h + 44, "640 × 480 pixel, 60 volte al secondo"),
  );
}

function prepareDiagram(): SVGSVGElement {
  return canvas(
    "Da 640 per 480 a 416 per 416 pixel, e da 212 a 53",
    svg("defs", {}, spectrum("att-prep")),
    svg("rect", { x: 24, y: 96, width: 232, height: 174, rx: 10, fill: "url(#att-prep)", class: "d-image" }),
    grid(24, 96, 232, 174, 14.5),
    svg("rect", { x: 24, y: 96, width: 232, height: 174, rx: 10, class: "d-outline" }),
    label(140, 300, "640 × 480"),
    svg("path", { d: "M276 183 H340 M326 169 L342 183 L326 197", class: "d-arrow" }),
    svg("rect", { x: 362, y: 64, width: 238, height: 238, rx: 10, class: "d-pad" }),
    svg(
      "g",
      { class: "d-shrink" },
      svg("rect", { x: 362, y: 94, width: 238, height: 178, fill: "url(#att-prep)", class: "d-image" }),
      grid(362, 94, 238, 178, 11.9),
    ),
    svg("rect", { x: 362, y: 64, width: 238, height: 238, rx: 10, class: "d-outline" }),
    label(481, 332, "416 × 416"),
    svg("rect", { x: 132, y: 372, width: 128, height: 72, rx: 14, class: "d-tile" }),
    label(196, 420, "212", "d-label d-big"),
    label(196, 466, "da 0 a 255", "d-label d-small"),
    svg("path", { d: "M276 408 H346 M332 394 L348 408 L332 422", class: "d-arrow" }),
    label(311, 390, "÷ 4", "d-label d-small"),
    svg("rect", { x: 364, y: 372, width: 128, height: 72, rx: 14, class: "d-tile d-tile-int" }),
    label(428, 420, "53", "d-label d-big d-int"),
    label(428, 466, "intero a 8 bit", "d-label d-small"),
  );
}

function chipDiagram(): SVGSVGElement {
  const cells: SVGElement[] = [];
  for (let row = 0; row < 10; row += 1) {
    for (let col = 0; col < 10; col += 1) {
      cells.push(
        svg("rect", {
          x: 205 + col * 24,
          y: 110 + row * 24,
          width: 18,
          height: 18,
          rx: 3,
          class: "d-mac",
          style: `animation-delay: ${(row + col) * 70}ms`,
        }),
      );
    }
  }
  const pins: string[] = [];
  for (let index = 0; index < 9; index += 1) {
    const offset = 112 + index * 27;
    pins.push(`M${offset} 70V86M${offset} 394V410`, `M150 ${offset + 18}H166M474 ${offset + 18}H490`);
  }
  return canvas(
    "Il chip della scheda: una griglia di moltiplicatori che lavorano insieme",
    svg("path", { d: pins.join(""), class: "d-pin" }),
    svg("rect", { x: 166, y: 86, width: 308, height: 308, rx: 22, class: "d-chip" }),
    ...cells,
    label(320, 40, "AMD Kria KV260"),
    label(320, 448, "DPU: migliaia di moltiplicazioni nello stesso istante", "d-label d-small"),
  );
}

/** COCO-17 keypoints of a standing figure, in diagram units. */
const SKELETON: ReadonlyArray<[number, number]> = [
  [560, 318], [553, 312], [567, 312], [546, 316], [574, 316],
  [540, 340], [580, 340], [532, 372], [588, 372], [528, 402], [592, 402],
  [548, 400], [572, 400], [546, 432], [574, 432], [545, 462], [575, 462],
];
const BONES: ReadonlyArray<[number, number]> = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

function networkDiagram(): SVGSVGElement {
  const stacks = [
    { x: 150, size: 120, step: 7.5, top: "52 × 52", name: "bordi", delay: 0 },
    { x: 320, size: 92, step: 11.5, top: "26 × 26", name: "forme", delay: 600 },
    { x: 468, size: 66, step: 16.5, top: "13 × 13", name: "oggetti", delay: 1200 },
  ];
  const parts: SVGElement[] = [
    svg("defs", {}, spectrum("att-net")),
    svg("rect", { x: 14, y: 150, width: 92, height: 92, rx: 8, fill: "url(#att-net)", class: "d-image" }),
    grid(14, 150, 92, 92, 11.5),
    svg("rect", { x: 14, y: 150, width: 92, height: 92, rx: 8, class: "d-outline" }),
    label(60, 276, "foto"),
  ];
  for (const stack of stacks) {
    const y = 196 - stack.size / 2;
    const layers: SVGElement[] = [];
    for (let depth = 2; depth >= 0; depth -= 1) {
      const offset = depth * 10;
      layers.push(
        svg("rect", {
          x: stack.x + offset,
          y: y - offset,
          width: stack.size,
          height: stack.size,
          rx: 6,
          class: depth === 0 ? "d-layer d-layer-front" : "d-layer",
        }),
      );
    }
    layers.push(grid(stack.x, y, stack.size, stack.size, stack.step));
    parts.push(
      svg("g", { class: "d-stage", style: `animation-delay: ${stack.delay}ms` }, ...layers),
      label(stack.x + stack.size / 2 + 10, y - 34, stack.top, "d-label d-small"),
      label(stack.x + stack.size / 2, 196 + stack.size / 2 + 34, stack.name),
    );
  }
  parts.push(
    svg("path", { d: "M110 196H146M276 196H316M418 196H464", class: "d-flow" }),
    // The detector's answer: one box with its class and confidence.
    svg("rect", { x: 40, y: 326, width: 200, height: 104, rx: 10, class: "d-panel" }),
    svg("rect", { x: 92, y: 356, width: 96, height: 62, rx: 4, class: "d-box" }),
    svg("rect", { x: 92, y: 336, width: 96, height: 22, rx: 4, class: "d-tag" }),
    label(140, 352, "tazza 92%", "d-label d-tag-text"),
    label(140, 460, "YOLOX-nano: 80 categorie", "d-label d-small"),
    // MoveNet: the 17 keypoints of a body, scaled into the bottom row.
    svg(
      "g",
      { transform: "translate(470 318) scale(0.75) translate(-560 -312)" },
      svg("path", {
        d: BONES.map(
          ([a, b]) => `M${SKELETON[a][0]} ${SKELETON[a][1]}L${SKELETON[b][0]} ${SKELETON[b][1]}`,
        ).join(""),
        class: "d-bone",
      }),
      ...SKELETON.map(([x, y]) => svg("circle", { cx: x, cy: y, r: 5, class: "d-joint" })),
    ),
    label(470, 460, "MoveNet: 17 punti del corpo", "d-label d-small", "middle"),
  );
  return canvas("Strati di una rete neurale: bordi, forme, oggetti", ...parts);
}

function cup(x: number, y: number): SVGElement {
  return svg(
    "g",
    { class: "d-cup" },
    svg("path", { d: `M${x} ${y}h70l-8 84h-54z`, class: "d-cup-body" }),
    svg("path", { d: `M${x + 68} ${y + 16}c26 0 26 40 -4 40`, class: "d-cup-handle" }),
  );
}

function trackingDiagram(): SVGSVGElement {
  return canvas(
    "Riquadri che tremano senza tracciamento, fermi con il tracciamento",
    svg("rect", { x: 14, y: 70, width: 296, height: 320, rx: 16, class: "d-panel" }),
    svg("rect", { x: 330, y: 70, width: 296, height: 320, rx: 16, class: "d-panel" }),
    label(162, 50, "senza tracciamento"),
    label(478, 50, "con tracciamento"),
    svg(
      "g",
      { class: "d-move" },
      cup(70, 200),
      svg("rect", { x: 56, y: 186, width: 112, height: 112, rx: 4, class: "d-box d-jitter" }),
    ),
    svg(
      "g",
      { class: "d-move" },
      cup(386, 200),
      svg("rect", { x: 372, y: 186, width: 112, height: 112, rx: 4, class: "d-box d-steady" }),
    ),
    label(162, 428, "il riquadro salta a ogni fotogramma", "d-label d-small"),
    label(478, 428, "lo stesso oggetto, lo stesso riquadro", "d-label d-small"),
  );
}

function mindDiagram(): SVGSVGElement {
  const nodes = [
    { x: 80, name: "Percezione", icon: "M58 220c12-16 32-16 44 0c-12 16-32 16-44 0zM80 213a7 7 0 1 0 0.1 0" },
    { x: 240, name: "Memoria breve", icon: "M218 206h44M218 220h44M218 234h30" },
    { x: 400, name: "Regole", icon: "M400 244V224M400 224L384 202M400 224L416 202" },
    { x: 560, name: "Voce", icon: "M542 214v12M552 206v28M562 198v44M572 208v24" },
  ];
  const parts: SVGElement[] = [
    svg("path", { d: "M80 220H560", class: "d-wire" }),
    svg("path", { d: "M560 274C560 400 80 400 80 274", class: "d-loop" }),
  ];
  nodes.forEach((node, index) => {
    parts.push(
      svg(
        "g",
        { class: "d-node", style: `animation-delay: ${index * 750}ms` },
        svg("circle", { cx: node.x, cy: 220, r: 50, class: "d-node-ring" }),
        svg("path", { d: node.icon, class: "d-node-icon" }),
      ),
      label(node.x, 306, node.name),
    );
  });
  parts.push(
    svg("circle", { cx: 80, cy: 220, r: 9, class: "d-pulse" }),
    // What the board sends is only the input: the mind runs on the laptop.
    svg("path", { d: "M80 92V160M68 148L80 162L92 148", class: "d-arrow" }),
    label(80, 70, "oggetti e pose dalla scheda", "d-label d-small", "start"),
    label(320, 456, "tutto questo gira sul portatile", "d-label d-small"),
  );
  return canvas("Percezione, memoria breve, regole e voce", ...parts);
}

function privacyDiagram(): SVGSVGElement {
  return canvas(
    "Le immagini vengono analizzate e dimenticate: niente cloud, niente salvataggi",
    svg("defs", {}, spectrum("att-priv")),
    svg(
      "g",
      { class: "d-forget" },
      svg("rect", { x: 40, y: 170, width: 160, height: 120, rx: 10, fill: "url(#att-priv)", class: "d-image" }),
      grid(40, 170, 160, 120, 16),
    ),
    svg("circle", { cx: 330, cy: 230, r: 74, class: "d-chip" }),
    grid(290, 190, 80, 80, 16),
    label(330, 340, "analisi"),
    // A cloud and a disk, each crossed out.
    svg("path", {
      d: "M498 150a22 22 0 0 1 42-10a18 18 0 0 1 34 14a16 16 0 0 1-4 32h-68a18 18 0 0 1-4-36z",
      class: "d-icon",
    }),
    svg("path", { d: "M488 112L590 204", class: "d-cross" }),
    label(540, 236, "nessun cloud", "d-label d-small"),
    svg("rect", { x: 502, y: 282, width: 76, height: 84, rx: 8, class: "d-icon" }),
    svg("path", { d: "M516 300h48M516 316h48", class: "d-icon" }),
    svg("path", { d: "M488 272L590 376", class: "d-cross" }),
    label(540, 404, "nessun salvataggio", "d-label d-small"),
  );
}

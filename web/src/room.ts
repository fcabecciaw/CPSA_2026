/**
 * The MoveNet room of the dashboard (R2-F): the 17-keypoint skeleton of
 * `pose.state` standing inside a stylised 3D room — perspective floor grid,
 * back wall with glowing edges, the avatar's cyan→violet palette.
 *
 * The skeleton is drawn on a plane in front of the back wall, mirrored like a
 * mirror image of the camera, smoothed so 30 Hz keypoints do not jitter, and
 * faded out when `pose.state.score` says nobody is in the frame (F-E4 #25 uses
 * the same 0.3 threshold for gestures).
 */

import * as THREE from "three";

import type { PoseStatePayload } from "./protocol";

/** COCO-17 bones of MoveNet. */
const BONES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [0, 5],
  [0, 6],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

const KEYPOINT_MIN_SCORE = 0.3;
const SMOOTHING = 12;
const MAX_PIXEL_RATIO = 1.5;
/** Where inside the room the skeleton plane sits (positive z is nearer). */
const SKELETON_Z = 0.45;

const TOP = new THREE.Color("#4fd6ff");
const MID = new THREE.Color("#7a6cff");
const BOTTOM = new THREE.Color("#ff4fd8");

export class RoomView {
  private readonly root: HTMLElement;
  private readonly renderer: THREE.WebGLRenderer | null;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 40);
  private readonly joints!: THREE.Points;
  private readonly bones!: THREE.LineSegments;
  private readonly jointPositions = new Float32Array(17 * 3);
  private readonly bonePositions = new Float32Array(BONES.length * 6);
  private readonly boneColors = new Float32Array(BONES.length * 6);
  private readonly smoothed = Array.from({ length: 17 }, () => new THREE.Vector3());
  private readonly targets = Array.from({ length: 17 }, () => new THREE.Vector3());
  private readonly seen = new Float32Array(17);
  private readonly jointColors = new Float32Array(17 * 3);
  private intensity = 0;
  private visible = false;
  private disposed = false;
  private frameHandle = 0;
  private lastAt: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.root.classList.add("room");
    this.renderer = createRenderer(root);
    if (this.renderer === null) {
      this.root.dataset.status = "nogl";
      return;
    }
    this.root.append(this.renderer.domElement);
    this.renderer.domElement.setAttribute("role", "img");
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Stanza 3D con lo scheletro rilevato da MoveNet",
    );

    this.camera.position.set(0, 1.28, 3.35);
    this.camera.lookAt(0, 1.02, 0);
    this.buildRoom();

    const jointGeometry = new THREE.BufferGeometry();
    jointGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.jointPositions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    jointGeometry.setAttribute("color", new THREE.BufferAttribute(this.jointColors, 3));
    this.joints = new THREE.Points(
      jointGeometry,
      new THREE.PointsMaterial({
        size: 0.11,
        map: spriteTexture(),
        vertexColors: true,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
      }),
    );
    this.joints.frustumCulled = false;

    const boneGeometry = new THREE.BufferGeometry();
    boneGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.bonePositions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    boneGeometry.setAttribute("color", new THREE.BufferAttribute(this.boneColors, 3));
    this.bones = new THREE.LineSegments(
      boneGeometry,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.bones.frustumCulled = false;

    this.scene.add(this.joints, this.bones);
    this.buildPalette();
    this.resize();
    this.resizeObserver =
      typeof ResizeObserver === "function" ? new ResizeObserver(() => this.resize()) : null;
    this.resizeObserver?.observe(this.root);
    document.addEventListener("visibilitychange", this.onVisibility);
  }

  /** Show the room; hidden keeps the WebGL loop idle. */
  setVisible(visible: boolean): void {
    if (visible === this.visible) {
      return;
    }
    this.visible = visible;
    this.root.hidden = !visible;
    if (visible) {
      this.lastAt = null;
      this.start();
      this.resize();
    } else {
      window.cancelAnimationFrame(this.frameHandle);
      this.frameHandle = 0;
    }
  }

  clear(): void {
    this.intensity = 0;
    this.applyOpacity(0);
  }

  /** One `pose.state` frame: map, smooth and store the 17 keypoints. */
  update(pose: PoseStatePayload, now: number): void {
    const dt = this.lastAt === null ? 1 / 30 : Math.min(0.2, (now - this.lastAt) / 1_000);
    this.lastAt = now;
    const blend = 1 - Math.exp(-dt * SMOOTHING);
    const width = Math.max(1, pose.width);
    const height = Math.max(1, pose.height);

    for (let index = 0; index < 17; index += 1) {
      const point = pose.keypoints[index];
      const score = point?.[2] ?? 0;
      const target = this.targets[index];
      if (score >= KEYPOINT_MIN_SCORE) {
        // The camera looks at the person, the room mirrors it: a wave of the
        // right hand is on the right of the skeleton too.
        target.set(
          -(point[0] / width - 0.5) * 2.3,
          (1 - point[1] / height) * 1.92 + 0.08,
          SKELETON_Z,
        );
        if (this.seen[index] === 0) {
          this.smoothed[index].copy(target);
        } else {
          this.smoothed[index].lerp(target, blend);
        }
        this.seen[index] = 1;
      } else {
        // A keypoint that drops out keeps its last place and fades, so the
        // skeleton never snaps to the floor.
        this.seen[index] = Math.max(0, this.seen[index] - dt * 3.5);
      }
    }

    const wanted = pose.score < KEYPOINT_MIN_SCORE ? 0 : Math.min(1, (pose.score - 0.2) / 0.35);
    this.intensity += (wanted - this.intensity) * (1 - Math.exp(-dt * 6));
    this.syncBuffers();
    this.applyOpacity(this.intensity);
  }

  private syncBuffers(): void {
    for (let index = 0; index < 17; index += 1) {
      const point = this.smoothed[index];
      this.jointPositions[index * 3] = point.x;
      this.jointPositions[index * 3 + 1] = point.y;
      this.jointPositions[index * 3 + 2] = point.z;
      const seen = this.seen[index] * 1.35;
      this.jointColors[index * 3] = TOP.r * seen;
      this.jointColors[index * 3 + 1] = TOP.g * seen;
      this.jointColors[index * 3 + 2] = TOP.b * seen;
    }
    for (let bone = 0; bone < BONES.length; bone += 1) {
      const [from, to] = BONES[bone];
      const a = this.smoothed[from];
      const b = this.smoothed[to];
      const at = bone * 6;
      this.bonePositions[at] = a.x;
      this.bonePositions[at + 1] = a.y;
      this.bonePositions[at + 2] = a.z;
      this.bonePositions[at + 3] = b.x;
      this.bonePositions[at + 4] = b.y;
      this.bonePositions[at + 5] = b.z;
      const fade = Math.min(this.seen[from], this.seen[to]) * 1.5;
      colorAt(a.y, BONE_FROM);
      colorAt(b.y, BONE_TO);
      this.boneColors[at] = BONE_FROM.r * fade;
      this.boneColors[at + 1] = BONE_FROM.g * fade;
      this.boneColors[at + 2] = BONE_FROM.b * fade;
      this.boneColors[at + 3] = BONE_TO.r * fade;
      this.boneColors[at + 4] = BONE_TO.g * fade;
      this.boneColors[at + 5] = BONE_TO.b * fade;
    }
    const jointGeometry = this.joints.geometry;
    jointGeometry.getAttribute("position").needsUpdate = true;
    jointGeometry.getAttribute("color").needsUpdate = true;
    const boneGeometry = this.bones.geometry;
    boneGeometry.getAttribute("position").needsUpdate = true;
    boneGeometry.getAttribute("color").needsUpdate = true;
  }

  private applyOpacity(intensity: number): void {
    const clamped = Math.max(0, Math.min(1, intensity));
    (this.joints.material as THREE.PointsMaterial).opacity = clamped;
    (this.bones.material as THREE.LineBasicMaterial).opacity = clamped * 0.85;
    this.joints.visible = clamped > 0.01;
    this.bones.visible = clamped > 0.01;
  }

  private buildRoom(): void {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(5.2, 5.2),
      new THREE.MeshBasicMaterial({ color: 0x050d16, transparent: true, opacity: 0.92 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(4.8, 16, 0x2f9dbf, 0x123a4c);
    grid.position.y = 0.002;
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const material of gridMaterials) {
      material.transparent = true;
      material.opacity = 0.55;
      material.depthWrite = false;
      (material as THREE.Material & { blending?: THREE.Blending }).blending =
        THREE.AdditiveBlending;
    }
    this.scene.add(grid);

    const wall = new THREE.Mesh(
      new THREE.PlaneGeometry(5.2, 2.9),
      new THREE.MeshBasicMaterial({ color: 0x060f1a, transparent: true, opacity: 0.95 }),
    );
    wall.position.set(0, 1.45, -2.6);
    this.scene.add(wall);

    const wallGrid = new THREE.GridHelper(5.2, 12, 0x1f6f8d, 0x102f3f);
    wallGrid.position.set(0, 1.45, -2.58);
    wallGrid.rotation.x = Math.PI / 2;
    const wallMaterials = Array.isArray(wallGrid.material)
      ? wallGrid.material
      : [wallGrid.material];
    for (const material of wallMaterials) {
      material.transparent = true;
      material.opacity = 0.3;
      material.depthWrite = false;
    }
    this.scene.add(wallGrid);

    const halo = new THREE.Mesh(
      new THREE.PlaneGeometry(5.4, 3.1),
      new THREE.MeshBasicMaterial({
        map: haloTexture(),
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    halo.position.set(0, 1.5, -2.55);
    this.scene.add(halo);

    // Glowing edge: back wall frame plus the floor meets, cyan at the top and
    // magenta where the room touches the ground.
    const edges: number[] = [];
    const colors: number[] = [];
    const corners: Array<[THREE.Vector3, THREE.Vector3]> = [
      [new THREE.Vector3(-2.6, 0, -2.6), new THREE.Vector3(-2.6, 2.9, -2.6)],
      [new THREE.Vector3(2.6, 0, -2.6), new THREE.Vector3(2.6, 2.9, -2.6)],
      [new THREE.Vector3(-2.6, 2.9, -2.6), new THREE.Vector3(2.6, 2.9, -2.6)],
      [new THREE.Vector3(-2.6, 0, -2.6), new THREE.Vector3(-2.6, 0, 2.2)],
      [new THREE.Vector3(2.6, 0, -2.6), new THREE.Vector3(2.6, 0, 2.2)],
      [new THREE.Vector3(-2.6, 0, 2.2), new THREE.Vector3(2.6, 0, 2.2)],
    ];
    for (const [from, to] of corners) {
      edges.push(from.x, from.y, from.z, to.x, to.y, to.z);
      for (const point of [from, to]) {
        colorAt(point.y / 2.9 - 0.02, BONE_FROM);
        colors.push(BONE_FROM.r, BONE_FROM.g, BONE_FROM.b);
      }
    }
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(edges, 3));
    edgeGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const edgeLines = new THREE.LineSegments(
      edgeGeometry,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.55,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    this.scene.add(edgeLines);
  }

  private buildPalette(): void {
    for (let index = 0; index < 17; index += 1) {
      this.jointColors[index * 3] = TOP.r;
      this.jointColors[index * 3 + 1] = TOP.g;
      this.jointColors[index * 3 + 2] = TOP.b;
    }
  }

  private start(): void {
    if (this.frameHandle !== 0) {
      return;
    }
    const loop = (): void => {
      if (!this.visible || this.disposed) {
        this.frameHandle = 0;
        return;
      }
      this.frameHandle = window.requestAnimationFrame(loop);
      if (document.hidden || this.renderer === null) {
        return;
      }
      this.renderer.render(this.scene, this.camera);
    };
    this.frameHandle = window.requestAnimationFrame(loop);
  }

  private resize(): void {
    if (this.renderer === null) {
      return;
    }
    const width = Math.max(1, this.root.clientWidth);
    const height = Math.max(1, this.root.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private readonly onVisibility = (): void => {
    this.lastAt = null;
    if (!document.hidden && this.visible) {
      this.start();
    }
  };

  dispose(): void {
    this.disposed = true;
    window.cancelAnimationFrame(this.frameHandle);
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.resizeObserver?.disconnect();
    this.joints.geometry.dispose();
    this.bones.geometry.dispose();
    (this.joints.material as THREE.Material).dispose();
    (this.bones.material as THREE.Material).dispose();
    this.renderer?.dispose();
    this.renderer?.domElement.remove();
  }
}

const BONE_FROM = new THREE.Color();
const BONE_TO = new THREE.Color();

/** Cyan at the top of the room, violet in the middle, magenta at the floor. */
function colorAt(height: number, out: THREE.Color): THREE.Color {
  const t = Math.max(0, Math.min(1, height));
  if (t < 0.5) {
    return out.copy(BOTTOM).lerp(MID, t * 2);
  }
  return out.copy(MID).lerp(TOP, (t - 0.5) * 2);
}

/** Soft vertical glow behind the skeleton, the room's only "light". */
function haloTexture(): THREE.Texture {
  const width = 256;
  const height = 256;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (context !== null) {
    const gradient = context.createRadialGradient(
      width / 2,
      height * 0.62,
      0,
      width / 2,
      height * 0.62,
      width * 0.5,
    );
    gradient.addColorStop(0, "rgba(122, 108, 255, 0.55)");
    gradient.addColorStop(0.45, "rgba(79, 214, 255, 0.22)");
    gradient.addColorStop(1, "rgba(4, 8, 15, 0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function spriteTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (context !== null) {
    const gradient = context.createRadialGradient(
      size / 2,
      size / 2,
      0,
      size / 2,
      size / 2,
      size / 2,
    );
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(0.35, "rgba(255,255,255,0.75)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function createRenderer(root: HTMLElement): THREE.WebGLRenderer | null {
  try {
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.className = "room-canvas";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";
    root.dataset.status = "ready";
    return renderer;
  } catch (error) {
    console.warn("[cpsa] stanza: WebGL non disponibile", error);
    return null;
  }
}

/**
 * The voice energy of the particle sphere (R2-F, round 3).
 *
 * The sphere speaks by pulsing: the amplitude of its surface follows the loudness
 * of the Italian voice. `attachAudio` taps a WebAudio node (the local TTS voice
 * exposes its `AnalyserNode` through `voiceAudioNode()` in `web/src/voice.ts`)
 * and measures the RMS with a fast attack and a slower release, which is what
 * makes each syllable a visible swell. Web Speech has no audio graph, so
 * `setLevel` takes an explicit level and, failing that, a small procedural
 * envelope keeps the sphere alive while it talks. The analyser is a passive
 * branch: attaching never changes what is heard.
 */

const FFT_SIZE = 512;

export interface VoiceEnergyOptions {
  /** Minimum level while speaking, so a quiet consonant still moves the cloud. */
  floor?: number;
  /** Level rise per second. */
  attack?: number;
  /** Level fall per second. */
  release?: number;
}

export class VoiceEnergy {
  private readonly floor: number;
  private readonly attack: number;
  private readonly release: number;
  private analyser: AnalyserNode | null = null;
  private waveform: Uint8Array<ArrayBuffer> | null = null;
  private externalLevel = 0;
  private level = 0;
  private startedAt = 0;

  constructor(options: VoiceEnergyOptions = {}) {
    this.floor = options.floor ?? 0.1;
    this.attack = options.attack ?? 22;
    this.release = options.release ?? 8;
  }

  get attached(): boolean {
    return this.analyser !== null;
  }

  /**
   * Tap an audio node. When the node already is an `AnalyserNode` (the voice
   * service's), it is read directly; any other node is branched through a small
   * analyser of our own. Returns the level measured on the next `update`.
   */
  attachAudio(node: AudioNode): void {
    this.detachAudio();
    if (isAnalyser(node)) {
      this.analyser = node;
      this.waveform = new Uint8Array(new ArrayBuffer(node.fftSize));
      return;
    }
    const analyser = node.context.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = 0.5;
    node.connect(analyser);
    this.analyser = analyser;
    this.waveform = new Uint8Array(new ArrayBuffer(analyser.fftSize));
  }

  detachAudio(): void {
    if (this.analyser !== null) {
      try {
        this.analyser.disconnect();
      } catch {
        // A closed audio context is not an error here.
      }
    }
    this.analyser = null;
    this.waveform = null;
  }

  /** Explicit level 0..1 (0 = silent), used by the screenshot hooks. */
  setLevel(level: number): void {
    this.externalLevel = clamp01(level);
  }

  /** Explicit level 0..1 for engines without a WebAudio graph (Web Speech). */
  setSpeakingLevel(level: number): void {
    this.setLevel(level);
  }

  /** One frame: smooth the measured (or fallback) level toward the voice. */
  update(now: number, deltaSeconds: number, speaking: boolean): number {
    const dt = Math.min(0.1, Math.max(0, deltaSeconds));
    let measured = 0;
    if (speaking) {
      const rms = this.measure();
      measured = Math.max(rms, this.externalLevel > 0.01 ? this.externalLevel : 0, this.floor);
      if (this.analyser === null && this.externalLevel <= 0.01) {
        measured = Math.max(measured, this.envelope(now));
      }
    }
    const rate = measured > this.level ? this.attack : this.release;
    const blend = 1 - Math.exp(-dt * rate);
    this.level += (measured - this.level) * blend;
    if (!speaking && this.level < 0.01) {
      this.level = 0;
      this.startedAt = now;
    }
    return this.level;
  }

  private measure(): number {
    const analyser = this.analyser;
    const waveform = this.waveform;
    if (analyser === null || waveform === null) {
      return 0;
    }
    analyser.getByteTimeDomainData(waveform);
    let sum = 0;
    for (let index = 0; index < waveform.length; index += 1) {
      const sample = (waveform[index] - 128) / 128;
      sum += sample * sample;
    }
    const rms = Math.sqrt(sum / waveform.length);
    // Speech sits around 0.05-0.25 RMS; the curve lifts normal speech to 1.
    return Math.min(1, Math.pow(rms * 3.4, 0.85) * 1.25);
  }

  /** A cheap syllable envelope: three detuned sines, no phoneme claim. */
  private envelope(now: number): number {
    if (this.startedAt === 0) {
      this.startedAt = now;
    }
    const seconds = (now - this.startedAt) / 1_000;
    const syllable = 0.5 + 0.5 * Math.sin(seconds * 15.7 + Math.sin(seconds * 5.3) * 2.1);
    const phrase = 0.6 + 0.4 * Math.sin(seconds * 1.9);
    return 0.15 + 0.55 * syllable * phrase;
  }
}

function isAnalyser(node: AudioNode): node is AnalyserNode {
  return typeof AnalyserNode !== "undefined" && node instanceof AnalyserNode;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

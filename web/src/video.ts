/**
 * F14: the camera pane of the dashboard.
 *
 * The server publishes the analysed frames on `/video.mjpeg`
 * (`multipart/x-mixed-replace`); the `<img>` keeps the last frame it decoded, so
 * the pane never flickers while a mode switches. A stream that fails keeps the
 * placeholder and retries with a bounded backoff instead of giving up.
 */

import { CAMERA_LABELS } from "./labels";

/** A video pane the dashboard can start and stop; the mock provides its own. */
export interface VideoSource {
  start(): void;
  stop(): void;
}

const RETRY_MIN_MS = 1_000;
const RETRY_MAX_MS = 10_000;

export class VideoFeed implements VideoSource {
  private readonly image: HTMLImageElement;
  private readonly note: HTMLElement;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private retryMs = RETRY_MIN_MS;
  private attempt = 0;
  private active = false;

  constructor(image: HTMLImageElement, note: HTMLElement) {
    this.image = image;
    this.note = note;
    this.image.addEventListener("load", this.onLoad);
    this.image.addEventListener("error", this.onError);
  }

  /** Opens the stream; a no-op while it already runs. */
  start(): void {
    if (this.active) {
      return;
    }
    this.active = true;
    this.retryMs = RETRY_MIN_MS;
    this.connect();
  }

  stop(): void {
    this.active = false;
    this.clearRetry();
    this.image.removeAttribute("src");
    this.image.hidden = true;
    this.showNote(CAMERA_LABELS.waiting);
  }

  private connect(): void {
    this.attempt += 1;
    // The parameter makes the browser open a new stream instead of reusing
    // the failed response from its cache.
    this.image.src = `/video.mjpeg?stream=${this.attempt}`;
  }

  private readonly onLoad = (): void => {
    this.retryMs = RETRY_MIN_MS;
    this.image.hidden = false;
    this.note.hidden = true;
  };

  private readonly onError = (): void => {
    this.image.hidden = true;
    this.showNote(CAMERA_LABELS.unavailable);
    this.scheduleRetry();
  };

  private scheduleRetry(): void {
    if (!this.active || this.retryTimer !== null) {
      return;
    }
    const delay = this.retryMs;
    this.retryMs = Math.min(this.retryMs * 2, RETRY_MAX_MS);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      if (this.active) {
        this.connect();
      }
    }, delay);
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  private showNote(text: string): void {
    this.note.textContent = text;
    this.note.hidden = false;
  }
}

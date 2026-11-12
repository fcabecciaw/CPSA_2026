/** Presentation by default, debug on request, the choice remembered. */

import { describe, expect, it } from "vitest";

import { memoryStorage } from "./cognitive/testkit";
import {
  UI_MODE_KEY,
  isToggleShortcut,
  readUiMode,
  writeUiMode,
  type ShortcutEvent,
} from "./uimode";
import type { StorageLike } from "./voice";

function key(overrides: Partial<ShortcutEvent> = {}): ShortcutEvent {
  return {
    key: "d",
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    repeat: false,
    target: null,
    ...overrides,
  };
}

const throwing: StorageLike = {
  getItem: () => {
    throw new Error("blocked");
  },
  setItem: () => {
    throw new Error("blocked");
  },
  removeItem: () => {
    throw new Error("blocked");
  },
};

describe("UI mode", () => {
  it("starts in presentation mode when nothing is stored", () => {
    expect(readUiMode(memoryStorage(), new URLSearchParams())).toBe("presentation");
    expect(readUiMode(null, new URLSearchParams())).toBe("presentation");
  });

  it("remembers the debug choice", () => {
    const storage = memoryStorage();
    writeUiMode(storage, "debug");
    expect(storage.getItem(UI_MODE_KEY)).toBe("debug");
    expect(readUiMode(storage, new URLSearchParams())).toBe("debug");
    writeUiMode(storage, "presentation");
    expect(readUiMode(storage, new URLSearchParams())).toBe("presentation");
  });

  it("lets ?debug= override the stored choice for one load", () => {
    const storage = memoryStorage();
    writeUiMode(storage, "debug");
    expect(readUiMode(storage, new URLSearchParams("debug=0"))).toBe("presentation");
    expect(readUiMode(memoryStorage(), new URLSearchParams("debug=1"))).toBe("debug");
  });

  it("survives a blocked storage", () => {
    expect(readUiMode(throwing, new URLSearchParams())).toBe("presentation");
    expect(() => writeUiMode(throwing, "debug")).not.toThrow();
  });
});

describe("the d shortcut", () => {
  it("toggles on a bare d or D", () => {
    expect(isToggleShortcut(key())).toBe(true);
    expect(isToggleShortcut(key({ key: "D" }))).toBe(true);
  });

  it("ignores other keys, modifiers and auto-repeat", () => {
    expect(isToggleShortcut(key({ key: "e" }))).toBe(false);
    expect(isToggleShortcut(key({ ctrlKey: true }))).toBe(false);
    expect(isToggleShortcut(key({ metaKey: true }))).toBe(false);
    expect(isToggleShortcut(key({ altKey: true }))).toBe(false);
    expect(isToggleShortcut(key({ repeat: true }))).toBe(false);
  });

  it("never fires while typing in a field", () => {
    const input = { tagName: "INPUT" } as unknown as EventTarget;
    const select = { tagName: "select" } as unknown as EventTarget;
    const editable = { tagName: "DIV", isContentEditable: true } as unknown as EventTarget;
    const button = { tagName: "BUTTON" } as unknown as EventTarget;
    expect(isToggleShortcut(key({ target: input }))).toBe(false);
    expect(isToggleShortcut(key({ target: select }))).toBe(false);
    expect(isToggleShortcut(key({ target: editable }))).toBe(false);
    expect(isToggleShortcut(key({ target: button }))).toBe(true);
  });
});

/**
 * Build-time switches of the booth UI.
 *
 * `VITE_SHOW_BRAND_LINE=1 npm run build` shows "EOLAB, Università di
 * Cagliari" under the title. It stays hidden until the wording is confirmed.
 */
export const SHOW_BRAND_LINE: boolean = import.meta.env.VITE_SHOW_BRAND_LINE === "1";

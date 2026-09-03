// Type shim for the vite-aliased vendored import (see vite.config.ts): the
// runtime comes from vendor/scriptor/vue/src, but type-checking
// their sources under our tsconfig is not our job - a narrow surface here is.
declare module '@truespar/scriptor-vue' {
  import type { DefineComponent } from 'vue'

  /** What ScriptorDoc exposes to a host that draws its own chrome - which the
   *  document pane does, so its toolbar sits beside lector's rather than
   *  scriptor bringing a second one. Zoom is a method pair rather than a prop
   *  because the core owns zoom too (Ctrl/Cmd +/-/0 and Ctrl+wheel); read it
   *  back on `state-change`. */
  /** Word's "Display for Review". scriptor's engine defaults to `all`, so a
   *  document with revisions shows them with no host involvement. */
  export type TrackDisplay = 'all' | 'simple' | 'none' | 'original'

  export interface ScriptorDocApi {
    loadDocx(bytes: Uint8Array): void
    toDocumentXml(): string
    /** 1 = 100%; the core clamps to 25%..400%. */
    setZoom(factor: number): void
    getZoom(): number
    /** 0 before the first render. */
    pageCount(): number
    trackDisplay(): TrackDisplay
    setTrackDisplay(mode: TrackDisplay): void
    /** Empty when the document carries no tracked changes or comments. */
    reviewers(): { name: string; color: string; visible: boolean }[]
  }

  export const ScriptorDoc: DefineComponent<
    {
      docx?: Uint8Array
      mode?: string
      gutter?: string
      selectable?: boolean
      /** zoom / page count / word count moved - refresh a host readout */
      onStateChange?: () => void
      onReady?: () => void
    },
    ScriptorDocApi
  >
}

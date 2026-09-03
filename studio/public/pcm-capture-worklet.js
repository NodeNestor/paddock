// Raw microphone frames for the realtime transcription session.
//
// MediaRecorder - what the Transcribe page's file recorder uses - produces a
// CONTAINER (webm/opus), which is the right thing when the answer is a file to
// post. A live session needs the opposite: uncompressed samples, continuously,
// with no container framing to wait for. That is what an AudioWorklet gives,
// and it is the only API that does: ScriptProcessorNode is deprecated and runs
// on the main thread, where a busy render blocks the microphone. (The
// composable still falls back to ScriptProcessorNode on plain-http origins,
// where `audioWorklet` is secure-context-gated and does not exist at all.)
//
// This runs in the audio rendering thread, so it does the least possible work:
// copy the block and post it. Rate conversion and PCM16 packing happen on the
// main thread, where being a millisecond late costs nothing.
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const block = inputs[0] && inputs[0][0]
    // A disconnected or silent-by-routing input hands us nothing; returning
    // true keeps the node alive so unmuting resumes rather than needing a
    // rebuild.
    if (block && block.length) this.port.postMessage(new Float32Array(block))
    return true
  }
}

registerProcessor('pcm-capture', PcmCapture)

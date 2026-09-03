// Minimal, robust Server-Sent-Events reader over a fetch ReadableStream.
//
// The OpenAI/Anthropic streaming endpoints return `text/event-stream`: frames
// separated by a blank line, each with one or more `data:` lines. We buffer
// across network chunks (a frame can split mid-chunk), yield each frame's
// joined `data` payload, and let the caller JSON.parse and interpret it.

/** Yield each SSE frame's `data` payload as a string, in order. */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // Frames are delimited by a blank line (\n\n or \r\n\r\n).
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() ?? '' // last element is an incomplete frame
      for (const frame of frames) {
        const data = frameData(frame)
        if (data !== null) yield data
      }
    }
    // flush a trailing frame with no terminating blank line
    const data = frameData(buffer)
    if (data !== null) yield data
  } finally {
    reader.releaseLock()
  }
}

/** Join the `data:` lines of one frame; ignore comments and other fields. */
function frameData(frame: string): string | null {
  const parts: string[] = []
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith('data:')) {
      // one optional leading space after the colon, per the SSE spec
      parts.push(line.slice(5).replace(/^ /, ''))
    }
  }
  return parts.length ? parts.join('\n') : null
}

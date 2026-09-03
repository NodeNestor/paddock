// Sigma node program with a ripple/radar pulse - how a query result says
// "look HERE" on the graph without moving the camera.
//
// Lifted from traverse studio (studio/src/programs/NodePulseProgram.ts @
// c1aaee4, itself adapted from ormeo.lens - WebGL shaders identical). Do not
// hand-improve; re-lift from upstream. The rAF driver that advances
// `currentTime` lives in GraphCanvas - the program only reads it.
import { NodeProgram } from 'sigma/rendering'
import { floatColor } from 'sigma/utils'
import type { NodeDisplayData, RenderParams } from 'sigma/types'

const VERTEX_SHADER = `
attribute vec4 a_id;
attribute vec4 a_color;
attribute vec2 a_position;
attribute float a_size;

uniform float u_sizeRatio;
uniform float u_pixelRatio;
uniform mat3 u_matrix;
uniform float u_time;

varying vec4 v_color;
varying float v_border;

const float bias = 255.0 / 254.0;
const float RIPPLE_MAX_SCALE = 2.5;

void main() {
  gl_Position = vec4(
    (u_matrix * vec3(a_position, 1)).xy,
    0,
    1
  );

  gl_PointSize = a_size / u_sizeRatio * u_pixelRatio * 2.0 * RIPPLE_MAX_SCALE;
  v_border = (0.5 / a_size) * u_sizeRatio;

  #ifdef PICKING_MODE
  v_color = a_id;
  #else
  v_color = a_color;
  #endif

  v_color.a *= bias;
}
`

const FRAGMENT_SHADER = `
precision highp float;

varying vec4 v_color;
varying float v_border;

uniform float u_time;

const float RIPPLE_MAX_SCALE = 2.5;
const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

void main(void) {
  vec2 m = gl_PointCoord - vec2(0.5, 0.5);
  float distFromCenter = length(m);

  if (distFromCenter > 0.5) discard;

  float nodeRadius = 0.5 / RIPPLE_MAX_SCALE;

  #ifdef PICKING_MODE
  if (distFromCenter < nodeRadius) {
    gl_FragColor = v_color;
  } else {
    discard;
  }
  #else

  float scaledBorder = v_border / RIPPLE_MAX_SCALE;
  float distToNodeEdge = nodeRadius - distFromCenter;

  float nodeT = 0.0;
  if (distToNodeEdge > scaledBorder) {
    nodeT = 1.0;
  } else if (distToNodeEdge > 0.0) {
    nodeT = distToNodeEdge / scaledBorder;
  }

  float rippleT = 0.0;

  if (distFromCenter > nodeRadius) {
    float outerDist = (distFromCenter - nodeRadius) / (0.5 - nodeRadius);
    float ripplePos = fract(u_time * 0.8);
    float ringWidth = 0.05;
    float diff = abs(outerDist - ripplePos);

    if (diff < ringWidth) {
      float ringIntensity = 1.0 - (diff / ringWidth);
      ringIntensity *= (1.0 - outerDist);
      rippleT = ringIntensity * 0.5;
    }
  }

  float finalT = max(nodeT, rippleT);
  if (finalT < 0.01) discard;

  gl_FragColor = mix(transparent, v_color, finalT);

  #endif
}
`

const UNIFORMS = ['u_sizeRatio', 'u_pixelRatio', 'u_matrix', 'u_time'] as const

class NodePulseProgram extends NodeProgram {
  static currentTime = 0

  getDefinition() {
    return {
      VERTICES: 1,
      VERTEX_SHADER_SOURCE: VERTEX_SHADER,
      FRAGMENT_SHADER_SOURCE: FRAGMENT_SHADER,
      METHOD: WebGLRenderingContext.POINTS,
      UNIFORMS: UNIFORMS as unknown as string[],
      ATTRIBUTES: [
        { name: 'a_position', size: 2, type: WebGLRenderingContext.FLOAT },
        { name: 'a_size', size: 1, type: WebGLRenderingContext.FLOAT },
        { name: 'a_color', size: 4, type: WebGLRenderingContext.UNSIGNED_BYTE, normalized: true },
        { name: 'a_id', size: 4, type: WebGLRenderingContext.UNSIGNED_BYTE, normalized: true },
      ],
    }
  }

  processVisibleItem(nodeIndex: number, startIndex: number, data: NodeDisplayData) {
    const array = this.array
    array[startIndex++] = data.x
    array[startIndex++] = data.y
    array[startIndex++] = data.size
    array[startIndex++] = floatColor(data.color)
    array[startIndex++] = nodeIndex
  }

  setUniforms(
    params: RenderParams,
    programInfo: {
      gl: WebGLRenderingContext
      uniformLocations: Record<string, WebGLUniformLocation>
    },
  ) {
    const { gl, uniformLocations } = programInfo
    gl.uniform1f(uniformLocations.u_sizeRatio, params.sizeRatio)
    gl.uniform1f(uniformLocations.u_pixelRatio, params.pixelRatio)
    gl.uniformMatrix3fv(uniformLocations.u_matrix, false, params.matrix)
    gl.uniform1f(uniformLocations.u_time, NodePulseProgram.currentTime)
  }
}

export default NodePulseProgram

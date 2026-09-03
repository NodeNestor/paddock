// Sigma node program drawing hexagons - the alternate node shape styling and
// GDS surfaces can assign (type: 'hexagon').
//
// Lifted from traverse studio (studio/src/programs/NodeHexagonProgram.ts @
// c1aaee4, itself adapted from ormeo.lens). Do not hand-improve; re-lift.
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

varying vec4 v_color;
varying float v_border;

const float bias = 255.0 / 254.0;

void main() {
  gl_Position = vec4(
    (u_matrix * vec3(a_position, 1)).xy,
    0,
    1
  );

  gl_PointSize = a_size / u_sizeRatio * u_pixelRatio * 2.0;
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
precision mediump float;

varying vec4 v_color;
varying float v_border;

const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);

float hexagonSDF(vec2 p, float r) {
  const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
  p = abs(p);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}

void main(void) {
  vec2 m = gl_PointCoord - vec2(0.5, 0.5);
  vec2 p = m * 2.0;

  float hexRadius = 0.9;
  float dist = hexagonSDF(vec2(p.y, p.x), hexRadius);

  #ifdef PICKING_MODE
  if (dist < 0.0)
    gl_FragColor = v_color;
  else
    discard;

  #else
  float t = 0.0;
  if (dist < -v_border)
    t = 1.0;
  else if (dist < 0.0)
    t = -dist / v_border;

  gl_FragColor = mix(transparent, v_color, t);
  #endif
}
`

const UNIFORMS = ['u_sizeRatio', 'u_pixelRatio', 'u_matrix'] as const

class NodeHexagonProgram extends NodeProgram {
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

  setUniforms(params: RenderParams, programInfo: { gl: WebGLRenderingContext; uniformLocations: Record<string, WebGLUniformLocation> }) {
    const { gl, uniformLocations } = programInfo
    gl.uniform1f(uniformLocations.u_sizeRatio, params.sizeRatio)
    gl.uniform1f(uniformLocations.u_pixelRatio, params.pixelRatio)
    gl.uniformMatrix3fv(uniformLocations.u_matrix, false, params.matrix)
  }
}

export default NodeHexagonProgram

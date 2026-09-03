// D3-force layout for a graphology graph rendered by sigma.
//
// Lifted from traverse studio (studio/src/utils/d3ForceLayout.ts @ c1aaee4,
// itself adapted from ormeo.lens) - do not hand-improve; if the upstream copy
// gets better, re-lift it.
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  forceX,
  forceY,
} from 'd3-force'
import type Graph from 'graphology'
import type Sigma from 'sigma'

export const D3_FORCE_DEFAULTS = {
  alpha: 1,
  alphaMin: 0.001,
  alphaDecay: 0.0228,
  alphaTarget: 0,
  linkDistance: 30,
  linkStrength: null as number | null,
  manyBodyStrength: -100,
  manyBodyTheta: 0.9,
  manyBodyDistanceMin: 1,
  manyBodyDistanceMax: 3000,
  collisionRadius: 15,
  collisionStrength: 0.7,
  centerX: 0,
  centerY: 0,
  positioningStrengthX: 0.1,
  positioningStrengthY: 0.1,
}

interface D3Node {
  id: string
  x: number
  y: number
  fx: number | null | undefined
  fy: number | null | undefined
  vx?: number
  vy?: number
  index?: number
  _attrs: Record<string, unknown>
}

interface D3Link {
  id: string
  source: D3Node
  target: D3Node
  index?: number
  _attrs: Record<string, unknown>
}

function graphToD3(graph: Graph): {
  nodes: D3Node[]
  links: D3Link[]
  nodeMap: Map<string, D3Node>
} {
  const nodes: D3Node[] = []
  const nodeMap = new Map<string, D3Node>()

  graph.forEachNode((id, attrs) => {
    const node: D3Node = {
      id,
      x: (attrs.x as number) ?? 0,
      y: (attrs.y as number) ?? 0,
      fx: attrs.fixed ? ((attrs.x as number) ?? 0) : null,
      fy: attrs.fixed ? ((attrs.y as number) ?? 0) : null,
      _attrs: attrs,
    }
    nodeMap.set(id, node)
    nodes.push(node)
  })

  const links: D3Link[] = []
  graph.forEachEdge((_id, attrs, source, target) => {
    const s = nodeMap.get(source)
    const t = nodeMap.get(target)
    if (s && t) {
      links.push({ id: _id, source: s, target: t, _attrs: attrs })
    }
  })

  return { nodes, links, nodeMap }
}

function applyPositionsToGraph(graph: Graph, nodes: D3Node[]): void {
  nodes.forEach((node) => {
    if (graph.hasNode(node.id)) {
      graph.setNodeAttribute(node.id, 'x', node.x)
      graph.setNodeAttribute(node.id, 'y', node.y)
    }
  })
}

function createSimulation(
  nodes: D3Node[],
  links: D3Link[],
  settings: Partial<typeof D3_FORCE_DEFAULTS> = {},
) {
  const config = { ...D3_FORCE_DEFAULTS, ...settings }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const simulation = forceSimulation(nodes as any)
    .alpha(config.alpha)
    .alphaMin(config.alphaMin)
    .alphaDecay(config.alphaDecay)
    .alphaTarget(config.alphaTarget)

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const linkForce = forceLink(links as any)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .id((d: any) => d.id)
    .distance(config.linkDistance)
  if (config.linkStrength !== null) linkForce.strength(config.linkStrength)
  simulation.force('link', linkForce)

  simulation.force(
    'charge',
    forceManyBody()
      .strength(config.manyBodyStrength)
      .theta(config.manyBodyTheta)
      .distanceMin(config.manyBodyDistanceMin)
      .distanceMax(config.manyBodyDistanceMax),
  )

  simulation.force(
    'collision',
    forceCollide().radius(config.collisionRadius).strength(config.collisionStrength),
  )
  simulation.force('center', forceCenter(config.centerX, config.centerY))
  simulation.force('x', forceX(config.centerX).strength(config.positioningStrengthX))
  simulation.force('y', forceY(config.centerY).strength(config.positioningStrengthY))

  return simulation
}

/** Run the simulation to convergence synchronously and write positions back. */
export function applyD3ForceLayout(
  graph: Graph,
  settings: Partial<typeof D3_FORCE_DEFAULTS> = {},
): Graph {
  const { nodes, links } = graphToD3(graph)
  const config = { ...D3_FORCE_DEFAULTS, ...settings }

  const simulation = createSimulation(nodes, links, config)
  simulation.stop()
  const iterations = Math.ceil(Math.log(config.alphaMin) / Math.log(1 - config.alphaDecay))

  for (let i = 0; i < iterations; i++) {
    simulation.tick()
  }

  applyPositionsToGraph(graph, nodes)
  return graph
}

/** Animated variant: ticks against a live sigma, converges or times out. */
export class D3ForceLayoutController {
  private graph: Graph | null
  private sigma: Sigma | null
  private settings: typeof D3_FORCE_DEFAULTS
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private simulation: any = null
  public nodes: D3Node[] | null = null
  private links: D3Link[] | null = null
  public isRunning = false
  private _checkInterval: ReturnType<typeof setInterval> | null = null
  private _onEnd: (() => void) | null = null

  constructor(graph: Graph, sigma: Sigma | null, settings: Partial<typeof D3_FORCE_DEFAULTS> = {}) {
    this.graph = graph
    this.sigma = sigma
    this.settings = { ...D3_FORCE_DEFAULTS, ...settings }
  }

  start(
    options: {
      onTick?: () => void
      onEnd?: () => void
      maxRuntime?: number
      convergenceThreshold?: number
      stableChecksRequired?: number
    } = {},
  ): void {
    const {
      onTick,
      onEnd,
      maxRuntime = 5000,
      convergenceThreshold = 0.1,
      stableChecksRequired = 3,
    } = options

    this.stop()
    this._onEnd = onEnd ?? null
    if (!this.graph) return

    const { nodes, links } = graphToD3(this.graph)
    this.nodes = nodes
    this.links = links

    this.simulation = createSimulation(nodes, links, this.settings)
    this.isRunning = true

    const startTime = Date.now()
    const previousPositions = new Map<string, { x: number; y: number }>()
    let stableChecks = 0

    nodes.forEach((n) => previousPositions.set(n.id, { x: n.x, y: n.y }))

    this.simulation.on('tick', () => {
      if (this.graph) applyPositionsToGraph(this.graph, this.nodes!)
      this.sigma?.refresh()
      onTick?.()
    })

    this.simulation.on('end', () => this._finish())

    this._checkInterval = setInterval(() => {
      if (!this.isRunning) {
        clearInterval(this._checkInterval!)
        return
      }
      if (Date.now() - startTime > maxRuntime) {
        this.stop()
        return
      }

      let totalMovement = 0
      let nodeCount = 0
      this.nodes!.forEach((n) => {
        const prev = previousPositions.get(n.id)
        if (prev) {
          totalMovement += Math.sqrt((n.x - prev.x) ** 2 + (n.y - prev.y) ** 2)
        }
        previousPositions.set(n.id, { x: n.x, y: n.y })
        nodeCount++
      })

      if (nodeCount > 0 && totalMovement / nodeCount < convergenceThreshold) {
        stableChecks++
        if (stableChecks >= stableChecksRequired) this.stop()
      } else {
        stableChecks = 0
      }
    }, 500)
  }

  stop(): void {
    if (this._checkInterval) {
      clearInterval(this._checkInterval)
      this._checkInterval = null
    }
    if (this.simulation) {
      this.simulation.stop()
      this.simulation = null
    }
    if (this.isRunning) {
      this.isRunning = false
      this._finish()
    }
  }

  kill(): void {
    this.stop()
    this.graph = null
    this.sigma = null
    this.nodes = null
    this.links = null
  }

  private _finish(): void {
    this._onEnd?.()
    this._onEnd = null
  }

  reheat(alpha = 0.3): void {
    if (this.simulation) {
      this.simulation.alpha(alpha).restart()
      this.isRunning = true
    }
  }

  fixNode(nodeId: string, x: number, y: number): void {
    const node = this.nodes?.find((n) => n.id === nodeId)
    if (node) {
      node.fx = x
      node.fy = y
    }
  }

  releaseNode(nodeId: string): void {
    const node = this.nodes?.find((n) => n.id === nodeId)
    if (node) {
      node.fx = null
      node.fy = null
    }
  }
}

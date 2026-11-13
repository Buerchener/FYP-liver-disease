export function element(tag, className = '', text = '') {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

export function edgeTone(edge) {
  if (edge.review === 'supported') return 'supported';
  if (['uncertain', 'unreviewed'].includes(edge.review)) return 'uncertain';
  return 'issue';
}

export function indexGraph(data) {
  const nodes = new Map(data.nodes.map(node => [node.id, {...node, parents: [], children: []}]));
  const names = new Map();
  data.nodes.forEach(node => names.set(node.name, (names.get(node.name) || 0) + 1));
  for (const node of nodes.values()) {
    node.label = names.get(node.name) > 1 ? node.name + ' · ' + node.id.slice(-6) : node.name;
  }
  for (const edge of data.edges) {
    nodes.get(edge.parent).children.push(edge);
    nodes.get(edge.child).parents.push(edge);
  }
  const scopes = new Map();
  for (const root of data.roots) {
    const seen = new Set();
    const pending = [root];
    while (pending.length) {
      const id = pending.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      pending.push(...nodes.get(id).children.map(edge => edge.child));
    }
    scopes.set(root, seen);
  }
  const rooted = new Set([...scopes.values()].flatMap(ids => [...ids]));
  scopes.set('unrooted', new Set([...nodes.keys()].filter(id => !rooted.has(id))));
  return {nodes, scopes};
}

export class TaxonomyGraph {
  constructor(container, nodes, onSelect, onEdge) {
    this.container = container;
    this.nodes = nodes;
    this.onSelect = onSelect;
    this.onEdge = onEdge;
    this.marks = new Map();
    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.setAttribute('class', 'graph-lines');
    this.svg.setAttribute('aria-hidden', 'true');
    const defs = document.createElementNS(this.svg.namespaceURI, 'defs');
    for (const tone of ['supported', 'uncertain', 'issue']) {
      const marker = document.createElementNS(this.svg.namespaceURI, 'marker');
      for (const [name, value] of Object.entries({
        id: 'arrow-' + tone, viewBox: '0 0 8 8', refX: 7, refY: 4,
        markerWidth: 6, markerHeight: 6, orient: 'auto', class: tone,
      })) marker.setAttribute(name, value);
      const arrow = document.createElementNS(this.svg.namespaceURI, 'path');
      arrow.setAttribute('d', 'M0 0L8 4L0 8Z');
      arrow.setAttribute('fill', 'currentColor');
      marker.append(arrow);
      defs.append(marker);
    }
    this.lines = document.createElementNS(this.svg.namespaceURI, 'g');
    this.svg.append(defs, this.lines);
    this.parents = element('div', 'graph-parents');
    this.focus = element('div', 'graph-focus');
    this.children = element('div', 'graph-children');
    container.replaceChildren(this.svg, this.parents, this.focus, this.children);
    this.observer = new ResizeObserver(() => this.schedule());
    this.observer.observe(container);
  }

  mark(id, selected = false) {
    const node = this.nodes.get(id);
    const mark = element(selected ? 'div' : 'button', 'concept-node');
    mark.dataset.kind = node.kind;
    mark.dataset.id = id;
    const shape = element('span', 'node-shape');
    shape.setAttribute('aria-hidden', 'true');
    mark.append(shape, element('span', '', node.label));
    if (selected) {
      mark.dataset.selected = '';
    } else {
      mark.type = 'button';
      mark.addEventListener('click', event => {
        this.onSelect(id);
        if (event.detail === 0) {
          (this.parents.querySelector('button') || this.children.querySelector('button'))?.focus({preventScroll: true});
        }
      });
    }
    this.marks.set(id, mark);
    return mark;
  }

  show(id) {
    this.current = this.nodes.get(id);
    this.marks.clear();
    const sort = endpoint => (a, b) => this.nodes.get(a[endpoint]).label.localeCompare(this.nodes.get(b[endpoint]).label);
    this.parentEdges = [...this.current.parents].sort(sort('parent'));
    this.childEdges = [...this.current.children].sort(sort('child'));
    this.parents.hidden = !this.parentEdges.length;
    this.children.hidden = !this.childEdges.length;
    this.parents.toggleAttribute('data-single', this.parentEdges.length === 1);
    this.children.toggleAttribute('data-single', this.childEdges.length === 1);
    this.parents.replaceChildren(...this.parentEdges.map(edge => this.mark(edge.parent)));
    this.focus.replaceChildren(this.mark(id, true));
    this.children.replaceChildren(...this.childEdges.map(edge => this.mark(edge.child)));
    this.schedule();
  }

  schedule() {
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => this.draw());
  }

  path(d, className) {
    const path = document.createElementNS(this.svg.namespaceURI, 'path');
    path.setAttribute('d', d);
    path.setAttribute('class', className);
    path.setAttribute('fill', 'none');
    this.lines.append(path);
    return path;
  }

  edgePath(d, edge) {
    const tone = edgeTone(edge);
    this.path(d, 'edge ' + tone).setAttribute('marker-end', 'url(#arrow-' + tone + ')');
    const hit = this.path(d, 'hit');
    hit.dataset.edge = edge.id;
    const title = document.createElementNS(this.svg.namespaceURI, 'title');
    title.textContent = edge.review_label + ' · ' + edge.review_reason;
    hit.append(title);
    hit.addEventListener('click', () => this.onEdge(edge));
  }

  draw() {
    if (!this.current || !this.container.clientWidth) return;
    const width = this.container.clientWidth;
    const count = this.childEdges.length;
    const columns = width >= 620 ? (count <= 3 ? Math.max(1, count) : 2) :
      (width >= 510 && count > 1 ? 2 : 1);
    const spread = count <= columns;
    const parentColumns = width >= 410 ? Math.max(1, this.parentEdges.length) : 1;
    this.parents.style.setProperty('--columns', parentColumns);
    this.parents.dataset.layout = this.parentEdges.length > parentColumns ? 'list' : 'spread';
    this.children.style.setProperty('--columns', columns);
    this.children.dataset.layout = spread ? 'spread' : 'list';
    this.lines.replaceChildren();
    const origin = this.container.getBoundingClientRect();
    const box = id => {
      const rect = this.marks.get(id).getBoundingClientRect();
      const left = rect.left - origin.left;
      const top = rect.top - origin.top;
      return {left, top, bottom: top + rect.height, x: left + rect.width / 2, y: top + rect.height / 2, width: rect.width};
    };
    const focus = box(this.current.id);
    this.parentEdges.forEach((edge, index) => {
      const parent = box(edge.parent);
      const x = focus.left + focus.width * (index + 1) / (this.parentEdges.length + 1);
      if (this.parentEdges.length > parentColumns) {
        const gutter = 5 + index * 8;
        const join = focus.top - 10 - index * 8;
        this.edgePath('M' + parent.left + ' ' + parent.y + 'H' + gutter + 'V' + join + 'H' + x + 'V' + (focus.top - 3), edge);
      } else {
        const bend = (parent.bottom + focus.top) / 2;
        this.edgePath('M' + parent.x + ' ' + parent.bottom + 'C' + parent.x + ' ' + bend + ' ' + x + ' ' + bend + ' ' + x + ' ' + (focus.top - 3), edge);
      }
    });
    if (!count) return;
    const childBoxes = this.childEdges.map(edge => box(edge.child));
    const join = (focus.bottom + Math.min(...childBoxes.map(rect => rect.top))) / 2;
    const gutter = rect => spread ? rect.x : rect.left - 16;
    const gutters = childBoxes.map(gutter);
    this.path('M' + focus.x + ' ' + focus.bottom + 'V' + join, 'trunk');
    this.path('M' + Math.min(focus.x, ...gutters) + ' ' + join + 'H' + Math.max(focus.x, ...gutters), 'trunk');
    if (!spread) {
      const bottoms = new Map();
      childBoxes.forEach(rect => bottoms.set(gutter(rect), Math.max(rect.y, bottoms.get(gutter(rect)) || 0)));
      for (const [x, y] of bottoms) this.path('M' + x + ' ' + join + 'V' + y, 'trunk');
    }
    this.childEdges.forEach((edge, index) => {
      const rect = childBoxes[index];
      this.edgePath(spread ? 'M' + rect.x + ' ' + join + 'V' + (rect.top - 3) :
        'M' + gutter(rect) + ' ' + rect.y + 'H' + (rect.left - 3), edge);
    });
  }
}

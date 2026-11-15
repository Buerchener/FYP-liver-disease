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

export function conceptButton(concept, onSelect) {
  const button = element('button', 'concept-node');
  const shape = element('span', 'node-shape');
  shape.setAttribute('aria-hidden', 'true');
  button.type = 'button';
  button.lang = 'en';
  button.dataset.id = concept.id;
  button.dataset.kind = concept.kind;
  button.append(shape, element('span', '', concept.label));
  button.addEventListener('click', () => onSelect(concept.id));
  return button;
}

// Top-down forest: independent roots remain independent, without an invented parent.
export class BranchGraph {
  constructor(container, nodes, onSelect, {overview = false} = {}) {
    this.container = container;
    this.nodes = nodes;
    this.onSelect = onSelect;
    this.overview = overview;
    this.marks = new Map();
    this.scale = 1;
    this.x = this.y = 0;
    container.classList.add('tree-canvas');
    const toolbar = element('div', 'tree-toolbar');
    const button = (label, action, title = label) => {
      const b = element('button', 'button', label);
      b.type = 'button'; b.title = title; b.setAttribute('aria-label', title);
      b.addEventListener('click', action); toolbar.append(b); return b;
    };
    button('−', () => this.zoom(1 / 1.2), '缩小');
    this.percent = element('span', 'tree-percent');
    toolbar.append(this.percent);
    button('+', () => this.zoom(1.2), '放大');
    button('适合宽度', () => this.fit(false));
    button('查看全图', () => this.fit(true));
    toolbar.append(element('span', 'tree-help', '拖动空白处移动 · 点击节点查看定义'));
    this.viewport = element('div', 'tree-viewport');
    this.viewport.tabIndex = 0;
    this.viewport.setAttribute('aria-label', '树状关系画布，可用方向键移动');
    this.world = element('div', 'tree-world');
    this.viewport.append(this.world);
    container.replaceChildren(toolbar, this.viewport);
    this.viewport.addEventListener('wheel', event => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const rect = this.viewport.getBoundingClientRect();
      this.zoom(Math.exp(-event.deltaY * .008), event.clientX - rect.left, event.clientY - rect.top);
    }, {passive: false});
    this.viewport.addEventListener('pointerdown', event => {
      if (event.target.closest('button') || event.button !== 0) return;
      this.drag = {x: event.clientX, y: event.clientY, ox: this.x, oy: this.y};
      this.viewport.setPointerCapture(event.pointerId);
      this.viewport.classList.add('dragging');
    });
    this.viewport.addEventListener('pointermove', event => {
      if (!this.drag) return;
      this.x = this.drag.ox + event.clientX - this.drag.x;
      this.y = this.drag.oy + event.clientY - this.drag.y;
      this.transform();
    });
    const end = () => { this.drag = null; this.viewport.classList.remove('dragging'); };
    this.viewport.addEventListener('pointerup', end);
    this.viewport.addEventListener('pointercancel', end);
    this.viewport.addEventListener('keydown', event => {
      if (event.target !== this.viewport) return;
      const delta = {ArrowLeft: [60, 0], ArrowRight: [-60, 0], ArrowUp: [0, 60], ArrowDown: [0, -60]}[event.key];
      if (delta) { event.preventDefault(); this.x += delta[0]; this.y += delta[1]; this.transform(); }
    });
    new ResizeObserver(() => {
      if (this.viewport.clientWidth && this.viewport.clientWidth !== this.lastWidth) {
        this.lastWidth = this.viewport.clientWidth; this.fit(false);
      }
    }).observe(this.viewport);
  }

  children(id) {
    return this.nodes.get(id).children.map(edge => edge.child).sort(
      (a, b) => this.nodes.get(a).label.localeCompare(this.nodes.get(b).label));
  }

  branchDepth(id) { return 1 + Math.max(0, ...this.children(id).map(child => this.branchDepth(child))); }

  show(root, selected) {
    const roots = Array.isArray(root) ? root : [root];
    const key = roots.join('|');
    if (key !== this.key) {
      this.key = key; this.marks.clear(); this.world.replaceChildren();
      const width = 180, height = 78, gap = 14, step = 118;
      const makeTree = (id, depth = 0) => {
        const kids = this.children(id).map(child => makeTree(child, depth + 1));
        return {id, depth, kids, width: Math.max(width, kids.reduce((sum, child) => sum + child.width, 0) + Math.max(0, kids.length - 1) * gap)};
      };
      const trees = roots.map(id => makeTree(id));
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.classList.add('tree-links'); svg.setAttribute('aria-hidden', 'true'); this.world.append(svg);
      const place = (tree, left, top) => {
        const x = left + tree.width / 2, y = top + tree.depth * step;
        const node = this.nodes.get(tree.id);
        const mark = conceptButton(node, this.onSelect);
        mark.title = node.label;
        mark.style.left = (x - width / 2) + 'px'; mark.style.top = y + 'px';
        mark.classList.add('tree-node');
        if (!node.children.length) mark.dataset.terminal = 'true';
        if (!node.parents.length) mark.dataset.treeRoot = 'true';
        this.world.append(mark);
        if (!this.marks.has(tree.id)) this.marks.set(tree.id, []);
        this.marks.get(tree.id).push(mark);
        let childLeft = left;
        for (const child of tree.kids) {
          const endX = childLeft + child.width / 2, endY = top + child.depth * step;
          const middleY = y + height + (step - height) / 2;
          const path = document.createElementNS(svg.namespaceURI, 'path');
          path.setAttribute('d', `M ${x} ${y + height} V ${middleY} H ${endX} V ${endY}`);
          path.dataset.parent = tree.id; path.dataset.child = child.id;
          svg.append(path); place(child, childLeft, top); childLeft += child.width + gap;
        }
      };
      const padding = 24, treeGap = 36;
      const rowWidth = row => row.reduce((sum, tree) => sum + tree.width, 0) + Math.max(0, row.length - 1) * treeGap;
      // Pack smaller trees into the same row instead of reserving two equal slots.
      const budget = this.overview ? Math.max(180, rowWidth(trees.slice(0, 2))) : (trees[0]?.width || 180);
      const rows = [];
      for (const tree of trees) {
        let row = rows[rows.length - 1];
        if (!row || rowWidth([...row, tree]) > budget) { row = []; rows.push(row); }
        row.push(tree);
      }
      this.width = Math.max(228, ...rows.map(row => rowWidth(row) + padding * 2));
      let y = padding;
      for (const row of rows) {
        let x = (this.width - rowWidth(row)) / 2;
        for (const tree of row) { place(tree, x, y); x += tree.width + treeGap; }
        y += (Math.max(...row.map(tree => this.branchDepth(tree.id))) - 1) * step + height + treeGap;
      }
      this.height = Math.max(126, y - treeGap + padding);
      this.world.style.width = this.width + 'px'; this.world.style.height = this.height + 'px';
      svg.setAttribute('width', this.width); svg.setAttribute('height', this.height);
      this.fit(false);
    }
    for (const [id, marks] of this.marks) for (const mark of marks) mark.setAttribute('aria-pressed', String(id === selected));
  }

  fit(all) {
    if (!this.width || !this.viewport.clientWidth) return;
    const available = this.viewport.clientWidth;
    const widthScale = Math.min(1, (available - 24) / this.width);
    const narrow = available < 600;
    // The overview grows vertically so labels remain legible rather than shrinking the entire forest.
    this.viewport.style.height = (this.overview ? Math.max(480, this.height * widthScale + 24) : 570) + 'px';
    this.scale = all ? Math.min(widthScale, (Math.min(680, window.innerHeight - 180) - 32) / this.height) : Math.max(narrow ? .8 : 0, widthScale);
    if (all) this.viewport.style.height = Math.max(400, Math.min(680, window.innerHeight - 180)) + 'px';
    this.x = (available - this.width * this.scale) / 2; this.y = 12;
    if (narrow && !all) {
      this.viewport.style.height = '570px';
      const first = this.world.querySelector('[data-tree-root]');
      if (first) this.x = available / 2 - (parseFloat(first.style.left) + 90) * this.scale;
    }
    this.transform();
  }

  zoom(factor, cx = this.viewport.clientWidth / 2, cy = Math.min(280, this.viewport.clientHeight / 2)) {
    const next = Math.max(.12, Math.min(2.5, this.scale * factor));
    this.x = cx - (cx - this.x) * next / this.scale;
    this.y = cy - (cy - this.y) * next / this.scale;
    this.scale = next; this.transform();
  }

  transform() {
    this.world.style.transform = `translate(${this.x}px, ${this.y}px) scale(${this.scale})`;
    this.percent.textContent = Math.round(this.scale * 100) + '%';
  }

  focus(id) { this.marks.get(id)?.[0]?.focus({preventScroll: true}); }
  schedule() { requestAnimationFrame(() => { if (!this.lastWidth) this.fit(false); }); }
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

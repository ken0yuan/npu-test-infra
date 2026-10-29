"""Source identities and bounded, explicit helper-to-test evidence.

This directory is advisory: runtime collection remains authoritative. Names,
path similarity and PR prose never establish a call or a regression assertion.
"""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from .test_suites import suite_for


def dotted(node):
    if isinstance(node, ast.Call):
        return dotted(node.func)
    if isinstance(node, ast.Subscript):  # triton_kernel[grid](...)
        return dotted(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return parent + "." + node.attr if parent else ""
    return ""


def own_nodes(node):
    """Do not attribute a nested function's body to its enclosing test."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from own_nodes(child)


@dataclass
class Object:
    name: str
    node: ast.AST
    owner: str
    kind: str
    start: int
    end: int


class Catalog:
    def __init__(self, text, path):
        import hashlib
        self.path = path
        self.normalized_source_sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.tree = ast.parse(text, filename=path)
        self.aliases = {}
        self.objects = {}
        self.classes = {}
        self.class_nodes = {}
        for node in own_nodes(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = ((node.module or '') + '.' + alias.name).strip('.')
        self._visit(self.tree.body, '', '')
        self.edges = {}
        self._focus_cache = {}
        self._reference_cache = {}

    def expand(self, name):
        head, sep, tail = name.partition('.')
        return self.aliases.get(head, head) + (sep + tail if sep else '')

    def _visit(self, body, scope, owner):
        for node in body:
            if isinstance(node, ast.ClassDef):
                name = '.'.join(x for x in (scope, node.name) if x)
                self.classes[name] = [self.expand(dotted(b)) for b in node.bases]
                self.class_nodes[name] = node
                self._visit(node.body, name, name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = '.'.join(x for x in (scope, node.name) if x)
                decorators = {self.expand(dotted(d)) for d in node.decorator_list}
                kernel = any(d in {'triton.jit', 'triton.jit.jit'} for d in decorators)
                nested = scope in self.objects
                kind = 'kernel' if kernel else 'helper'
                if not kernel and not nested and node.name.startswith('test'):
                    # TestCase classes may have arbitrary names. An unknown base
                    # stays a candidate until collection, rather than disappearing.
                    bases = self.classes.get(owner, [])
                    if not owner or owner.rsplit('.', 1)[-1].startswith('Test') or bases:
                        kind = 'test_candidate'
                    else:
                        # PyTorch copies methods from plain template classes.
                        # Keep their source identity pending runtime proof.
                        kind = 'template_candidate'
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                self.objects[name] = Object(name, node, owner, kind, start, node.end_lineno)
                self._visit(node.body, name, owner)
            else:
                for field in ('body', 'orelse', 'finalbody'):
                    children = getattr(node, field, None)
                    if isinstance(children, list):
                        self._visit(children, scope, owner)
                for handler in getattr(node, 'handlers', []):
                    self._visit(handler.body, scope, owner)

    @property
    def tests(self):
        return {n: (o.start, o.end) for n, o in self.objects.items()
                if o.kind in {'test_candidate', 'template_candidate'}}

    def _resolve_local(self, token, obj):
        method_call = token.startswith(('self.', 'cls.'))
        if method_call:
            token = obj.owner + '.' + token.split('.', 1)[1]
        if '.' in token and token in self.objects:
            return token
        scope = obj.name
        while scope:
            key = scope + '.' + token
            if scope not in self.classes and key in self.objects:
                return key
            scope = scope.rpartition('.')[0]
        if token in self.objects:
            return token
        # Inherited helpers, resolved only when the local base is unambiguous.
        if obj.owner and token.startswith(obj.owner + '.'):
            method = token.rsplit('.', 1)[-1]
            for base in self.classes.get(obj.owner, []):
                key = base + '.' + method
                if key in self.objects:
                    return key
        return None

    def _calls(self, obj):
        result = set()
        bound = {a.arg for a in (*obj.node.args.posonlyargs, *obj.node.args.args, *obj.node.args.kwonlyargs)}
        bound.update(n.id for n in own_nodes(obj.node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        for node in own_nodes(obj.node):
            if isinstance(node, ast.Call):
                token = dotted(node.func)
                if token.split('.', 1)[0] in bound and not token.startswith(('self.', 'cls.')):
                    continue
                target = self._resolve_local(token, obj)
                if target:
                    result.add(target)
        return result

    def reachable(self, test):
        queue = deque([(test, [test])]); seen = set()
        while queue:
            name, chain = queue.popleft()
            if name in seen:
                continue
            seen.add(name)
            yield name, chain
            if name not in self.edges:
                self.edges[name] = self._calls(self.objects[name])
            queue.extend((target, chain + [target]) for target in sorted(self.edges[name]) if target not in seen)

    def changed(self, lines, added=False):
        # Most specific definition wins; enclosing tests are not marked changed
        # just because a nested helper's implementation changed.
        hits = set()
        shared = set()
        for line in lines:
            matches = [o for o in self.objects.values() if o.start <= line <= o.end]
            if matches:
                hits.add(min(matches, key=lambda o: o.end - o.start).name)
            else:
                shared.add(line)
        if added:
            hits.update(self.tests)
        return hits, shared

    def changed_test_evidence(self, lines, added=False):
        changed, shared = self.changed(lines, added)
        changed_bindings = set()
        for node in own_nodes(self.tree):
            if getattr(node, 'lineno', None) not in shared:
                continue
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                changed_bindings.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                changed_bindings.update(a.asname or a.name.split('.')[0] for a in node.names)
        result = {}
        for test in self.tests:
            if test in changed:
                result[test] = dict(relevance=0, type='changed_test_method', chain=[test])
                continue
            for name, chain in self.reachable(test):
                if name in changed:
                    result[test] = dict(relevance=1, type='changed_helper_call', chain=chain)
                    break
                used = {n.id for n in own_nodes(self.objects[name].node)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} & changed_bindings
                if used:
                    result[test] = dict(relevance=2, type='changed_binding_reference', chain=chain+[sorted(used)[0]])
                    break
            if test not in result and shared:
                owner = self.objects[test].owner
                applicable = []
                for line in shared:
                    enclosing = [(name, node) for name, node in self.class_nodes.items()
                                 if min([node.lineno]+[d.lineno for d in node.decorator_list]) <= line <= node.end_lineno]
                    if enclosing:
                        scope = min(enclosing, key=lambda pair: pair[1].end_lineno-pair[1].lineno)[0]
                        if owner == scope or scope in self.classes.get(owner, []):
                            applicable.append(scope)
                    elif not changed_bindings:
                        applicable.append('<module>')
                if applicable:
                    result[test] = dict(relevance=2, type='shared_test_context', chain=[sorted(applicable)[0],test])
            if test not in result:
                # setup/fixture changes affect the appropriate class/module,
                # never pretend that every method directly calls that fixture.
                owner = self.objects[test].owner
                parameters = {a.arg for a in self.objects[test].node.args.args}
                fixtures = [name for name in changed if name in self.objects and
                            ((name.rsplit('.', 1)[-1] in {'setUp', 'setUpClass', 'setup_method', 'setup_class'}
                              and self.objects[name].owner == owner)
                             or (name in parameters and any(dotted(d).split('.')[-1] == 'fixture'
                                     for d in self.objects[name].node.decorator_list)))]
                if fixtures:
                    result[test] = dict(relevance=2, type='shared_fixture', chain=fixtures + [test])
        return result

    def references(self, test, changed_modules):
        refs = []
        signature = tuple(sorted((module, tuple(sorted(symbols))) for module, symbols in changed_modules.items()))
        for name, chain in self.reachable(test):
            key = (name, signature)
            if key in self._reference_cache:
                refs.extend(dict(ref, test_node=test, chain=chain) for ref in self._reference_cache[key])
                continue
            before = len(refs)
            obj = self.objects[name]
            calls = {dotted(n.func) for n in own_nodes(obj.node) if isinstance(n, ast.Call)}
            aliases = dict(self.aliases)
            bound = {a.arg for a in (*obj.node.args.posonlyargs, *obj.node.args.args, *obj.node.args.kwonlyargs)}
            for node in own_nodes(obj.node):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = ((node.module or '')+'.'+alias.name).strip('.')
            for node in own_nodes(obj.node):
                if not isinstance(node, (ast.Name, ast.Attribute)):
                    continue
                raw = dotted(node)
                head, sep, tail = raw.partition('.')
                if head in bound:
                    continue
                token = aliases.get(head, head) + (sep+tail if sep else '')
                for module, symbols in changed_modules.items():
                    if not token.startswith(module + '.'):
                        continue
                    symbol = token[len(module) + 1:].split('.')[0]
                    if symbol in symbols:
                        ref = dict(module=module, symbol=symbol, test_node=test,
                                   line=str(node.lineno), chain=chain,
                                   reference_kind='call' if dotted(node) in calls else 'reference')
                        if ref not in refs:
                            refs.append(ref)
            self._reference_cache[key] = [{k: v for k, v in ref.items() if k not in {'test_node', 'chain'}}
                                          for ref in refs[before:]]
        return refs

    def affected_exports(self, symbols, lines=()):
        """Reverse local calls: tests may call a public wrapper of a changed helper."""
        seeds, _ = self.changed(lines)
        if not lines:
            seeds.update(name for name in self.objects if name.rsplit('.', 1)[-1] in symbols)
        result = {symbol: [symbol] for symbol in symbols}
        for name in self.objects:
            for reached, chain in self.reachable(name):
                if reached in seeds:
                    export = name.split('.', 1)[0]
                    if export not in result or len(chain) < len(result[export]):
                        result[export] = chain
                    break
        return result

    def focus(self, lines):
        import re
        key = frozenset(lines)
        if key in self._focus_cache:
            return self._focus_cache[key]
        values = []
        for node in ast.walk(self.tree):
            if getattr(node, 'lineno', None) not in lines:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
                values.extend(re.findall(r'[a-z0-9]+', str(node.value).lower()))
            elif isinstance(node, ast.Attribute):
                values.append(node.attr.lower())
        result = sorted({v for v in values if len(v) >= 2})
        self._focus_cache[key] = result
        return result


def order_candidates(entries):
    """One strongest representative per affected group before redundant tests.

    Evidence levels are lexical tiers, never name/length/PR-mention scores.
    Stable greedy coverage is only compared within one tier.
    """
    entries = list(entries)
    if not entries or not any('relevance' in e for e in entries):
        return entries
    ordered = sorted(entries, key=lambda e: (e.get('relevance', 3), e['path'], e['test_nodes'][0]))
    rounds = {}; decorated = []
    for e in ordered:
        key = (e.get('relevance', 3), e['path'])
        turn = rounds.get(key, 0); rounds[key] = turn + 1
        decorated.append((key[0], turn, -len(e.get('groups', ())), e['path'], e['test_nodes'][0], e))
    ordered = [row[-1] for row in sorted(decorated, key=lambda row: row[:-1])]
    representatives, remainder, covered = [], [], set()
    for e in ordered:
        groups = set(e.get('groups', ()))
        if e.get('relevance', 3) <= 2 and groups - covered:
            representatives.append(e); covered.update(groups)
        else:
            remainder.append(e)
    return representatives + remainder


def sample_instances(nodes, limit, focus=()):
    """Prefer affected parameters, then spread over parameter/token strata."""
    import re
    nodes = sorted(set(nodes))
    if len(nodes) <= limit:
        return nodes
    focus = {str(x).lower() for x in focus if len(str(x)) >= 2}
    tokens = {n: set(re.findall(r'[a-z0-9]+', n.rsplit('::', 1)[-1].lower())) for n in nodes}
    selected = []; seen = set(); remaining = set(nodes)
    while remaining and len(selected) < limit:
        pick = min(remaining, key=lambda n: (-len(tokens[n] & focus),
                                            -len(tokens[n] - seen), n))
        selected.append(pick); seen.update(tokens[pick]); remaining.remove(pick)
    return selected

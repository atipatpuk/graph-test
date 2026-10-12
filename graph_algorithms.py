"""Python implementations of the graph algorithms used by Graph Theory Lab."""

from __future__ import annotations

from collections import deque
from html import escape
import math
from typing import Any


_PALETTE = [
    "#ef7d57", "#41a6f6", "#a7f070", "#ffcd75", "#f4a6d7",
    "#73eff7", "#b13e53", "#38b764", "#c28a4a", "#94b0c2",
    "#5d275d", "#3b5dc9", "#257179", "#d4a5ff", "#566c86",
]


def _fmt(value: float | int) -> str:
    if math.isinf(value):
        return "∞"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _name(names: dict[int, str], node_id: int) -> str:
    return names.get(node_id, f"v{node_id}")


def _html(value: Any) -> str:
    return escape(str(value), quote=True)


def _step(message: str, **state: dict[str, Any]) -> dict[str, Any]:
    return {
        "msg": message,
        "ntext": state.get("ntext"),
        "node": state.get("node", {}),
        "edge": state.get("edge", {}),
        "etext": state.get("etext"),
        "fill": state.get("fill"),
    }


def _card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div></div>'


def _result(title: str, key: str, steps: list[dict[str, Any]], summary: str, **extra: Any) -> dict[str, Any]:
    return {"title": title, "key": key, "steps": steps, "summary": summary, **extra}


def _graph(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str], bool]:
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("ข้อมูลกราฟไม่ถูกต้อง")
    node_ids: list[int] = []
    names: dict[int, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), int):
            raise ValueError("ข้อมูลจุดยอดไม่ถูกต้อง")
        node_id = node["id"]
        if node_id in names:
            raise ValueError("พบรหัสจุดยอดซ้ำ")
        node_ids.append(node_id)
        names[node_id] = str(node.get("label", f"v{node_id}"))
    valid_ids = set(node_ids)
    cleaned_edges: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("id"), int):
            raise ValueError("ข้อมูลเส้นเชื่อมไม่ถูกต้อง")
        u, v, weight = edge.get("u"), edge.get("v"), edge.get("w")
        if not isinstance(u, int) or not isinstance(v, int) or u not in valid_ids or v not in valid_ids:
            raise ValueError("เส้นเชื่อมอ้างถึงจุดยอดที่ไม่มีอยู่")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError("น้ำหนักหรือความจุของเส้นเชื่อมไม่ถูกต้อง")
        cleaned_edges.append({"id": edge["id"], "u": u, "v": v, "w": weight})
    return nodes, cleaned_edges, names, bool(payload.get("directed"))


def _dijkstra(payload: dict[str, Any]) -> dict[str, Any]:
    nodes, edges, names, directed = _graph(payload)
    ids = [node["id"] for node in nodes]
    source, sink = payload.get("source"), payload.get("sink")
    if not ids:
        raise ValueError("ยังไม่มีจุดยอด")
    if source not in names:
        raise ValueError('กรุณาเลือก "ต้นทาง" ก่อนรัน Dijkstra')
    negative = [edge for edge in edges if edge["w"] < 0]
    if negative:
        edges_list = ", ".join(f'{_name(names, edge["u"])}–{_name(names, edge["v"])}' for edge in negative)
        raise ValueError(f"Dijkstra ใช้ไม่ได้กับน้ำหนักติดลบ: {edges_list}")

    adj: dict[int, list[tuple[int, float, int]]] = {node_id: [] for node_id in ids}
    for edge in edges:
        adj[edge["u"]].append((edge["v"], edge["w"], edge["id"]))
        if not directed:
            adj[edge["v"]].append((edge["u"], edge["w"], edge["id"]))
    distances = {node_id: math.inf for node_id in ids}
    previous: dict[int, int | None] = {node_id: None for node_id in ids}
    previous_edge: dict[int, int] = {}
    done: set[int] = set()
    steps: list[dict[str, Any]] = []
    relax_count = 0
    distances[source] = 0

    def state() -> dict[str, Any]:
        return {
            "ntext": {node_id: _fmt(distances[node_id]) for node_id in ids},
            "node": {node_id: "visited" for node_id in done},
            "edge": {edge_id: "soft" for edge_id in previous_edge.values()},
        }

    def snap(message: str, node: dict[int, str] | None = None, edge: dict[int, str] | None = None) -> None:
        current = state()
        if node:
            current["node"].update(node)
        if edge:
            current["edge"].update(edge)
        steps.append(_step(message, **current))

    snap(f"เริ่มต้น: d({_name(names, source)}) = 0 และจุดอื่นทั้งหมด = ∞", {source: "current"})
    while True:
        available = [node_id for node_id in ids if node_id not in done and distances[node_id] < math.inf]
        if not available:
            remaining = [node_id for node_id in ids if node_id not in done]
            if remaining:
                labels = ", ".join(names[node_id] for node_id in remaining)
                snap(f"จุดที่เหลือ ({labels}) เข้าไม่ถึงจากต้นทาง → ระยะ = ∞")
            break
        current = min(available, key=lambda node_id: distances[node_id])
        best_distance = distances[current]
        done.add(current)
        snap(f"เลือก {_name(names, current)} ซึ่งมีระยะน้อยที่สุด ({_fmt(best_distance)}) ในจุดที่ยังไม่ปิด → ปิดจุดนี้ (ระยะนี้คือคำตอบสุดท้ายแล้ว)", {current: "current"})
        for target, weight, edge_id in adj[current]:
            if target in done:
                continue
            relax_count += 1
            candidate = distances[current] + weight
            old = distances[target]
            if candidate < old:
                distances[target] = candidate
                previous[target] = current
                previous_edge[target] = edge_id
                message = f"Relax {_name(names, current)} → {_name(names, target)}: {_fmt(distances[current])} + {_fmt(weight)} = {_fmt(candidate)} < {_fmt(old)} ✔ อัปเดต d({_name(names, target)}) = {_fmt(candidate)}"
                snap(message, {current: "current", target: "consider"}, {edge_id: "consider"})
            else:
                message = f"Relax {_name(names, current)} → {_name(names, target)}: {_fmt(distances[current])} + {_fmt(weight)} = {_fmt(candidate)} ≥ {_fmt(old)} ✘ ไม่เปลี่ยน"
                snap(message, {current: "current", target: "consider"}, {edge_id: "reject"})

    def path_to(target: int) -> list[int] | None:
        if math.isinf(distances[target]):
            return None
        path: list[int] = []
        current: int | None = target
        while current is not None:
            path.append(current)
            current = previous[current]
        return list(reversed(path))

    has_sink = sink in names
    path = path_to(sink) if has_sink else None
    path_edges: list[int] = []
    if path:
        current = sink
        while previous[current] is not None:
            path_edges.append(previous_edge[current])
            current = previous[current]
        snap(
            f"✔ เสร็จสิ้น! เส้นทางสั้นที่สุด: {' → '.join(_name(names, node_id) for node_id in path)} | ระยะรวม = {_fmt(distances[sink])}",
            {node_id: "path" for node_id in path},
            {edge_id: "path" for edge_id in path_edges},
        )
    elif has_sink:
        snap(f"✘ เสร็จสิ้น: ไม่มีเส้นทางจาก {_name(names, source)} ไป {_name(names, sink)} (ระยะ = ∞)")
    else:
        snap("✔ เสร็จสิ้น! ตัวเลขเหนือจุด = ระยะทางสั้นที่สุดจากต้นทาง (เลือก \"ปลายทาง\" เพื่อดูเส้นทาง)")

    summary = ""
    if has_sink:
        if path:
            route = " → ".join(f'<span class="chip">{_html(_name(names, node_id))}</span>' for node_id in path)
            summary += f'<div class="hero">🚚 ระยะทางสั้นที่สุดจาก <b>{_html(_name(names, source))}</b> ไป <b>{_html(_name(names, sink))}</b> = <span class="bignum">{_fmt(distances[sink])}</span><div class="route">{route}</div></div>'
        else:
            summary += f'<div class="hero bad">✘ ไปไม่ถึง {_html(_name(names, sink))} จาก {_html(_name(names, source))} (ระยะ = ∞)</div>'
    reachable = sum(not math.isinf(distances[node_id]) for node_id in ids)
    summary += '<div class="cards">' + _card("ต้นทาง", _html(_name(names, source))) + _card("ไปถึงได้", f"{reachable} / {len(nodes)} จุด") + _card("จำนวนครั้ง Relax", relax_count) + _card("จำนวนขั้นตอน", len(steps)) + "</div>"
    summary += '<h3>ระยะทางจากต้นทางไปทุกจุด</h3><div class="tw"><table class="tbl"><tr><th>จุดยอด</th><th>ระยะ</th><th>เส้นทาง</th></tr>'
    for node in sorted(nodes, key=lambda n: (distances[n["id"]], n["id"])):
        target = node["id"]
        path = path_to(target)
        route = " → ".join(_html(_name(names, path_id)) for path_id in path) if path else "เข้าไม่ถึง"
        summary += f'<tr><td class="l">{_html(names[target])}</td><td><b>{_fmt(distances[target])}</b></td><td class="l">{route}</td></tr>'
    summary += "</table></div><div class=\"note\">ใช้หาเส้นทางขนส่งที่สั้นที่สุดหรือถูกที่สุดจากคลังสินค้าไปยังร้านค้า/ลูกค้า ถ้าน้ำหนักคือ \"ค่าขนส่ง\" ผลลัพธ์จะเป็นเส้นทางที่ประหยัดที่สุด</div>"
    return _result("📊 สรุปผล Dijkstra — เส้นทางสั้นที่สุด", "dijkstra", steps, summary)


def _prim(payload: dict[str, Any]) -> dict[str, Any]:
    nodes, edges, names, directed = _graph(payload)
    if directed:
        raise ValueError("Prim ใช้ได้เฉพาะกราฟไม่มีทิศทาง")
    ids = [node["id"] for node in nodes]
    if not ids:
        raise ValueError("ยังไม่มีจุดยอด")
    source = payload.get("source")
    start = source if source in names else ids[0]
    in_tree: set[int] = set()
    selected: list[tuple[dict[str, Any], int, float]] = []
    roots: list[int] = []
    total: float = 0
    steps: list[dict[str, Any]] = []

    def base() -> dict[str, dict[int, str]]:
        return {
            "node": {node_id: "visited" for node_id in in_tree},
            "edge": {edge["id"]: "tree" for edge, _, _ in selected},
        }

    def snap(message: str, node: dict[int, str] | None = None, edge: dict[int, str] | None = None) -> None:
        current = base()
        if node:
            current["node"].update(node)
        if edge:
            current["edge"].update(edge)
        steps.append(_step(message, **current))

    order = [start, *(node_id for node_id in ids if node_id != start)]
    for root in order:
        if root in in_tree:
            continue
        tree_number = len(roots) + 1
        roots.append(root)
        in_tree.add(root)
        message = f"เริ่มต้นที่ {_name(names, root)} — ต้นไม้มี 1 จุด น้ำหนักรวม = 0" if tree_number == 1 else f"ไม่มีเส้นไปยังจุดที่เหลือแล้ว (กราฟไม่ Connected) → เริ่มต้นไม้ต้นที่ {tree_number} ที่ {_name(names, root)}"
        snap(message, {root: "current"})
        while True:
            candidates = [edge for edge in edges if edge["u"] != edge["v"] and ((edge["u"] in in_tree) != (edge["v"] in in_tree))]
            if not candidates:
                break
            best = min(candidates, key=lambda edge: edge["w"])
            edge_state = {edge["id"]: "consider" for edge in candidates}
            candidate_list = ", ".join(f'{_name(names, e["u"])}–{_name(names, e["v"])} ({_fmt(e["w"])})' for e in sorted(candidates, key=lambda e: e["w"])[:6])
            if len(candidates) > 6:
                candidate_list += ", ..."
            snap(f"ตัวเลือก {len(candidates)} เส้นที่เชื่อมต้นไม้กับจุดใหม่: {candidate_list}", edge=edge_state)
            new_node = best["v"] if best["u"] in in_tree else best["u"]
            in_tree.add(new_node)
            total += best["w"]
            selected.append((best, tree_number, total))
            snap(f"✔ เลือกเส้นต้นทุนต่ำสุด {_name(names, best['u'])}–{_name(names, best['v'])} = {_fmt(best['w'])} → เพิ่ม {_name(names, new_node)} | น้ำหนักรวม = {_fmt(total)}", {new_node: "current"}, {best["id"]: "path"})

    tree_count = len(roots)
    snap(f"✔ เสร็จสิ้น! Minimum Spanning Tree มี {len(selected)} เส้น น้ำหนักรวม = {_fmt(total)}" if tree_count == 1 else f"✔ เสร็จสิ้น! ได้ Spanning Forest {tree_count} ต้น ({len(selected)} เส้น) น้ำหนักรวม = {_fmt(total)}")
    label = "Minimum Spanning Tree" if tree_count == 1 else f"Spanning Forest ({tree_count} ต้น)"
    summary = f'<div class="hero {"warn" if tree_count > 1 else ""}">🌳 {label} น้ำหนักรวมต่ำสุด = <span class="bignum">{_fmt(total)}</span></div>'
    summary += '<div class="cards">' + _card("จุดเริ่ม", _html(_name(names, start))) + _card("จำนวนเส้นที่ใช้", f"{len(selected)} = |V| − {tree_count}") + _card("จำนวนต้นไม้", tree_count) + _card("เส้นที่ไม่ใช้", len(edges) - len(selected)) + "</div>"
    if tree_count > 1:
        roots_html = " ".join(f'<span class="chip">{_html(_name(names, root))}</span>' for root in roots)
        summary += f"<p>⚠ กราฟไม่ Connected จึงได้ป่า (Forest) — เริ่มแต่ละต้นที่: {roots_html}</p>"
    summary += '<h3>ลำดับเส้นที่เลือก</h3><div class="tw"><table class="tbl"><tr><th>#</th><th>ต้นที่</th><th>เส้น</th><th>น้ำหนัก</th><th>รวมสะสม</th></tr>'
    for index, (edge, tree_number, cumulative) in enumerate(selected, 1):
        summary += f'<tr><td>{index}</td><td>{tree_number}</td><td class="l">{_html(_name(names, edge["u"]))} – {_html(_name(names, edge["v"]))}</td><td>{_fmt(edge["w"])}</td><td><b>{_fmt(cumulative)}</b></td></tr>'
    summary += "</table></div><div class=\"note\">วางโครงข่ายถนน ท่อประปา หรือสายไฟให้ทุกจุดเชื่อมถึงกัน โดยใช้งบประมาณรวมน้อยที่สุด</div>"
    return _result("📊 สรุปผล Prim — Minimum Spanning Tree", "prim", steps, summary)


def _coloring(payload: dict[str, Any]) -> dict[str, Any]:
    nodes, edges, names, directed = _graph(payload)
    if directed:
        raise ValueError("Coloring ใช้ได้เฉพาะกราฟไม่มีทิศทาง")
    ids = [node["id"] for node in nodes]
    if not ids:
        raise ValueError("ยังไม่มีจุดยอด")
    neighbors: dict[int, set[int]] = {node_id: set() for node_id in ids}
    for edge in edges:
        if edge["u"] != edge["v"]:
            neighbors[edge["u"]].add(edge["v"])
            neighbors[edge["v"]].add(edge["u"])
    order = sorted(ids, key=lambda node_id: (-len(neighbors[node_id]), node_id))
    colors: dict[int, int] = {}
    color_names = payload.get("colorNames", [])
    def color_name(color: int) -> str:
        return color_names[color] if color < len(color_names) and color_names[color] else f"สี {color + 1}"

    def color_token(color: int) -> str:
        return f"{{{{COLOR_NAME_{color}}}}}"

    steps: list[dict[str, Any]] = []
    def snap(message: str, node: dict[int, str] | None = None, edge: dict[int, str] | None = None) -> None:
        steps.append(_step(message, fill=dict(colors), node=node or {}, edge=edge or {}))

    order_text = ", ".join(f"{_name(names, node_id)}({len(neighbors[node_id])})" for node_id in order)
    snap(f"Welsh–Powell: เรียงจุดตาม Degree มาก→น้อย: {order_text}")
    color = 0
    while len(colors) < len(order):
        snap(f'🎨 เริ่มใช้สีที่ {color + 1} "{color_name(color)}" — ไล่ตามลำดับ ใส่สีให้จุดที่ไม่ติดกับจุดสีนี้')
        for node_id in order:
            if node_id in colors:
                continue
            conflict = next((neighbor for neighbor in neighbors[node_id] if colors.get(neighbor) == color), None)
            if conflict is None:
                colors[node_id] = color
                snap(f'{_name(names, node_id)} ไม่ติดกับจุดที่ใช้ "{color_name(color)}" → ระบายสีนี้', {node_id: "current"})
            else:
                crossing = next((edge for edge in edges if (edge["u"] == node_id and edge["v"] == conflict) or (edge["v"] == node_id and edge["u"] == conflict)), None)
                snap(f'{_name(names, node_id)} ติดกับ {_name(names, conflict)} ที่ใช้ "{color_name(color)}" แล้ว → ข้ามไปก่อน', {node_id: "current"}, {crossing["id"]: "consider"} if crossing else {})
        color += 1
    color_count = color
    delta = max((len(neighbors[node_id]) for node_id in ids), default=0)

    exact: tuple[int, dict[int, int]] | None = None
    if len(ids) <= 15:
        exact_order = sorted(ids, key=lambda node_id: -len(neighbors[node_id]))
        exact_colors: dict[int, int] = {}
        def backtrack(index: int, limit: int, used: int) -> bool:
            if index == len(exact_order):
                return True
            node_id = exact_order[index]
            for candidate_color in range(min(limit, used + 1)):
                if all(exact_colors.get(neighbor) != candidate_color for neighbor in neighbors[node_id]):
                    exact_colors[node_id] = candidate_color
                    if backtrack(index + 1, limit, max(used, candidate_color + 1)):
                        return True
                    del exact_colors[node_id]
            return False
        for limit in range(1, len(ids) + 1):
            exact_colors.clear()
            if backtrack(0, limit, 0):
                exact = (limit, dict(exact_colors))
                break
    done_message = f"✔ เสร็จสิ้น! Greedy ใช้ {color_count} สี (ขอบเขตบน Δ+1 = {delta + 1})" + (f" | Chromatic Number χ(G) = {exact[0]}" if exact else "")
    snap(done_message)

    summary = f'<div class="hero">🎨 Welsh–Powell ใช้ <span class="bignum">{color_count}</span> สี' + (f' • ค่าที่ดีที่สุดจริง χ(G) = <b>{exact[0]}</b>' if exact else "") + "</div>"
    exact_value = str(exact[0]) if exact else "คำนวณเมื่อ |V| ≤ 15"
    summary += '<div class="cards">' + _card("Greedy (Welsh–Powell)", f"{color_count} สี") + _card("Δ (Degree สูงสุด)", delta) + _card("ขอบเขตบน Δ+1", delta + 1) + _card("χ(G) แบบ Exact", exact_value) + "</div>"
    if exact:
        summary += f"<p>⚠ Greedy ใช้สีมากกว่าค่าที่ดีที่สุด {color_count - exact[0]} สี (Greedy ไม่รับประกันว่าน้อยที่สุดเสมอ)</p>" if exact[0] < color_count else "<p>✔ Greedy ได้จำนวนสีเท่ากับค่าที่ดีที่สุด</p>"
    summary += '<h3>กลุ่มของแต่ละสี (กดตั้งชื่อความหมายได้)</h3>'
    for color in range(color_count):
        members = [node for node in nodes if colors[node["id"]] == color]
        chips = "".join(f'<span class="chip">{_html(node["label"])}</span>' for node in members)
        swatch = _PALETTE[color % len(_PALETTE)]
        summary += f'<div class="grp"><div class="gh"><span><span class="sw" style="background:{swatch}"></span>{color_token(color)} <small>({len(members)} จุด)</small></span><button type="button" class="btn mini" data-cname="{color}">✏️ ตั้งชื่อ</button></div>{chips}</div>'
    summary += '<h3>ลำดับ Welsh–Powell</h3><div class="tw"><table class="tbl"><tr><th>#</th><th>จุด</th><th>deg</th><th>สี</th></tr>'
    for index, node_id in enumerate(order, 1):
        summary += f'<tr><td>{index}</td><td class="l">{_html(_name(names, node_id))}</td><td>{len(neighbors[node_id])}</td><td class="l"><span class="sw" style="background:{_PALETTE[colors[node_id] % len(_PALETTE)]}"></span>{color_token(colors[node_id])}</td></tr>'
    summary += "</table></div><div class=\"note\">จุดที่เชื่อมกัน = ขัดแย้งกัน (เช่น วิชาที่มีนักเรียนสอบซ้อน หรือสินค้าที่ห้ามเก็บรวมกัน) จึงต้องได้สีต่างกัน แต่ละสีจึงหมายถึงช่วงเวลาสอบหรือช่องจัดเก็บ 1 ช่อง จำนวนสีจึงเท่ากับจำนวนช่วงเวลาหรือช่องที่ต้องใช้</div>"
    return _result("📊 สรุปผล Graph Coloring", "color", steps, summary, colors=colors)


def _flow(payload: dict[str, Any]) -> dict[str, Any]:
    nodes, edges, names, directed = _graph(payload)
    ids = [node["id"] for node in nodes]
    source, sink = payload.get("source"), payload.get("sink")
    if source not in names or sink not in names:
        raise ValueError("กรุณาเลือก Source และ Sink ก่อนรัน Max Flow")
    if source == sink:
        raise ValueError("Source และ Sink ต้องเป็นคนละจุด")
    if any(edge["w"] < 0 for edge in edges):
        raise ValueError("ความจุของเส้นเชื่อมต้องไม่ติดลบ")

    graph: dict[int, list[dict[str, Any]]] = {node_id: [] for node_id in ids}
    original_arcs: list[dict[str, Any]] = []
    def add_arc(u: int, v: int, capacity: float, edge_id: int, sign: int) -> None:
        forward: dict[str, Any] = {"to": v, "cap": capacity, "flow": 0, "eid": edge_id, "sign": sign}
        backward: dict[str, Any] = {"to": u, "cap": 0, "flow": 0, "eid": None, "sign": 0}
        forward["rev"] = backward
        backward["rev"] = forward
        graph[u].append(forward)
        graph[v].append(backward)
        original_arcs.append(forward)

    for edge in edges:
        if edge["u"] == edge["v"]:
            continue
        add_arc(edge["u"], edge["v"], edge["w"], edge["id"], 1)
        if not directed:
            add_arc(edge["v"], edge["u"], edge["w"], edge["id"], -1)

    def flow_of() -> dict[int, float]:
        flow = {edge["id"]: 0 for edge in edges}
        for arc in original_arcs:
            flow[arc["eid"]] += arc["sign"] * arc["flow"]
        return flow

    steps: list[dict[str, Any]] = []
    augmentations: list[tuple[list[int], float, bool]] = []
    total: float = 0

    def base() -> dict[str, Any]:
        flows = flow_of()
        edge_text: dict[int, str] = {}
        edge_state: dict[int, str] = {}
        for edge in edges:
            value = abs(flows[edge["id"]])
            edge_text[edge["id"]] = f"{_fmt(value)}/{_fmt(edge['w'])}"
            if value > 1e-9:
                edge_state[edge["id"]] = "sat" if value >= edge["w"] - 1e-9 else "flow"
        return {"etext": edge_text, "edge": edge_state}

    def snap(message: str, node: dict[int, str] | None = None, edge: dict[int, str] | None = None) -> None:
        current = base()
        if node:
            current["node"] = node
        if edge:
            current["edge"].update(edge)
        steps.append(_step(message, **current))

    snap(f"เริ่มต้น: Flow ทุกเส้น = 0 (แสดงเป็น flow/capacity) | Source = {_name(names, source)}, Sink = {_name(names, sink)}" + ("" if directed else " | ⚠ กราฟไม่มีทิศ: แต่ละเส้นไหลได้ 2 ทาง"), {source: "current", sink: "current"})
    for iteration in range(1, 10001):
        parent: dict[int, tuple[int, dict[str, Any]] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            current = queue.popleft()
            for arc in graph[current]:
                if arc["cap"] - arc["flow"] > 1e-9 and arc["to"] not in parent:
                    parent[arc["to"]] = (current, arc)
                    queue.append(arc["to"])
        if sink not in parent:
            break
        path_arcs: list[dict[str, Any]] = []
        path_nodes = [sink]
        current = sink
        bottleneck = math.inf
        while current != source:
            parent_item = parent[current]
            assert parent_item is not None
            previous, arc = parent_item
            path_arcs.append(arc)
            bottleneck = min(bottleneck, arc["cap"] - arc["flow"])
            current = previous
            path_nodes.append(current)
        path_arcs.reverse()
        path_nodes.reverse()
        edge_state = {arc["eid"] if arc["eid"] is not None else arc["rev"]["eid"]: "consider" for arc in path_arcs}
        node_state = {node_id: "consider" for node_id in path_nodes}
        uses_reverse = any(arc["eid"] is None for arc in path_arcs)
        path_text = " → ".join(_name(names, node_id) for node_id in path_nodes)
        snap(f"รอบ {iteration}: BFS พบ Augmenting Path {path_text} | คอขวดของเส้นทาง = {_fmt(bottleneck)}" + (" (มีการดันกระแสย้อนกลับผ่าน residual edge)" if uses_reverse else ""), node_state, edge_state)
        for arc in path_arcs:
            arc["flow"] += bottleneck
            arc["rev"]["flow"] -= bottleneck
        total += bottleneck
        augmentations.append((path_nodes, bottleneck, uses_reverse))
        snap(f"เพิ่ม Flow +{_fmt(bottleneck)} ตามเส้นทางนี้ → Flow รวม = {_fmt(total)}", node_state, {edge_id: "path" for edge_id in edge_state})

    reachable = {source}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for arc in graph[current]:
            if arc["cap"] - arc["flow"] > 1e-9 and arc["to"] not in reachable:
                reachable.add(arc["to"])
                queue.append(arc["to"])
    cut = [edge for edge in edges if (edge["u"] in reachable and edge["v"] not in reachable) or (not directed and edge["v"] in reachable and edge["u"] not in reachable)]
    cut_capacity = sum(edge["w"] for edge in cut)
    snap(f"ไม่พบ Augmenting Path อีกแล้ว → Max Flow = {_fmt(total)}")
    cut_names = ", ".join(f'{_name(names, edge["u"])}–{_name(names, edge["v"])}' for edge in cut) or "-"
    reachable_names = ", ".join(_name(names, node_id) for node_id in reachable)
    snap(f"✔ Min Cut: ฝั่ง S = {{{reachable_names}}} | เส้นคอขวด: {cut_names} | ความจุรวม = {_fmt(cut_capacity)} = Max Flow", {node_id: "inS" for node_id in reachable}, {edge["id"]: "cut" for edge in cut})

    final_flows = flow_of()
    cut_ids = {edge["id"] for edge in cut}
    summary = f'<div class="hero">🚰 ปริมาณไหลสูงสุดจาก <b>{_html(_name(names, source))}</b> ไป <b>{_html(_name(names, sink))}</b> = <span class="bignum">{_fmt(total)}</span></div>'
    summary += '<div class="cards">' + _card("จำนวนรอบ Augment", len(augmentations)) + _card("ความจุ Min Cut", _fmt(cut_capacity)) + _card("เส้นคอขวด", f"{len(cut)} เส้น") + _card("Max Flow = Min Cut", "✔ จริง" if math.isclose(total, cut_capacity, abs_tol=1e-9) else "✘") + "</div>"
    summary += '<h3>🔴 เส้นคอขวด (Min Cut)</h3>' + ("".join(f'<span class="chip" style="background:#ffd6d6">{_html(_name(names, edge["u"]))}–{_html(_name(names, edge["v"]))} ({_fmt(edge["w"])})</span>' for edge in cut) if cut else "<p>-</p>")
    side_s = " ".join(f'<span class="chip">{_html(_name(names, node_id))}</span>' for node_id in reachable)
    side_t = " ".join(f'<span class="chip">{_html(names[node_id])}</span>' for node_id in ids if node_id not in reachable)
    summary += f'<p style="font-size:14px">ฝั่ง S: {side_s}<br>ฝั่ง T: {side_t}</p>'
    summary += '<h3>Augmenting Paths ทีละรอบ</h3><div class="tw"><table class="tbl"><tr><th>รอบ</th><th>เส้นทาง</th><th>เพิ่ม</th></tr>'
    for index, (path, amount, uses_reverse) in enumerate(augmentations, 1):
        path_text = " → ".join(_html(_name(names, node_id)) for node_id in path) + (" ↺" if uses_reverse else "")
        summary += f"<tr><td>{index}</td><td class=\"l\">{path_text}</td><td><b>+{_fmt(amount)}</b></td></tr>"
    summary += "</table></div><h3>Flow บนแต่ละเส้น</h3><div class=\"tw\"><table class=\"tbl\"><tr><th>เส้น</th><th>Flow / Capacity</th><th>สถานะ</th></tr>"
    for edge in edges:
        value = final_flows[edge["id"]]
        absolute = abs(value)
        direction = f" ({_name(names, edge['u'])}→{_name(names, edge['v'])})" if not directed and absolute > 1e-9 and value > 0 else (f" ({_name(names, edge['v'])}→{_name(names, edge['u'])})" if not directed and absolute > 1e-9 else "")
        status = "🔴 คอขวด (เต็ม)" if edge["id"] in cut_ids else ("🟣 เต็ม" if absolute >= edge["w"] - 1e-9 and edge["w"] > 0 else f"เหลือ {_fmt(edge['w'] - absolute)}")
        summary += f'<tr><td class="l">{_html(_name(names, edge["u"]))}–{_html(_name(names, edge["v"]))}{_html(direction)}</td><td><b>{_fmt(absolute)}/{_fmt(edge["w"])}</b></td><td class="l">{status}</td></tr>'
    summary += "</table></div><div class=\"note\">ใช้ประเมินปริมาณสินค้าหรือการจราจรสูงสุดที่โครงข่ายรองรับได้ เส้นคอขวด (Min Cut) คือจุดที่ควรขยายก่อน เพราะถ้าเพิ่มความจุเส้นเหล่านี้ Max Flow จึงจะเพิ่มขึ้นได้</div>"
    return _result("📊 สรุปผล Max Flow (Edmonds–Karp)", "flow", steps, summary)


def run_algorithm(payload: dict[str, Any]) -> dict[str, Any]:
    algorithm = payload.get("algorithm")
    runners = {
        "dijkstra": _dijkstra,
        "prim": _prim,
        "color": _coloring,
        "flow": _flow,
    }
    if algorithm not in runners:
        raise ValueError("ไม่รู้จักอัลกอริทึมที่เลือก")
    return runners[algorithm](payload)

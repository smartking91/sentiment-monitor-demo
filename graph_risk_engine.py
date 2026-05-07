from collections import defaultdict, deque
from itertools import combinations
import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_companies(value):
    if pd.isna(value):
        return []
    parts = re.split(r"\s*\|\s*", str(value))
    seen = set()
    companies = []
    for item in parts:
        name = item.strip()
        if not name or name in seen:
            continue
        if len(name) < 2 or len(name) > 30:
            continue
        seen.add(name)
        companies.append(name)
    return companies


def _normalize_date(value):
    text = str(value)
    match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", text)
    if match:
        return pd.to_datetime(match.group(0), errors="coerce")
    return pd.NaT


def _build_negative_article_keys(df_news):
    if df_news is None or df_news.empty:
        return set()
    news = df_news.copy()
    if "sentiment" not in news.columns:
        return set()
    neg_mask = news["sentiment"].fillna("").astype(str).str.contains("负面|利空", regex=True, na=False)
    neg = news[neg_mask]
    keys = set()
    for _, row in neg.iterrows():
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip()
        if url:
            keys.add(("url", url))
        if title:
            keys.add(("title", title))
    return keys


def build_company_graph(news_with_companies, df_news=None, max_companies_per_article=16):
    """Build a company co-occurrence graph from article-level extracted companies."""
    if news_with_companies is None or news_with_companies.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = news_with_companies.copy()
    neg_keys = _build_negative_article_keys(df_news)
    latest_date = None
    if "datetime" in df.columns:
        latest_date = df["datetime"].apply(_normalize_date).max()
    if pd.isna(latest_date):
        latest_date = pd.Timestamp.today()

    node_stats = defaultdict(lambda: {"article_count": 0, "negative_article_count": 0})
    edge_stats = defaultdict(
        lambda: {
            "co_count": 0,
            "negative_co_count": 0,
            "time_decay_sum": 0.0,
            "sample_titles": [],
        }
    )

    for _, row in df.iterrows():
        companies = parse_companies(row.get("extracted_companies", ""))
        if len(companies) < 2:
            for company in companies:
                node_stats[company]["article_count"] += 1
            continue

        companies = companies[:max_companies_per_article]
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip()
        is_negative = (("url", url) in neg_keys) or (("title", title) in neg_keys)

        article_date = _normalize_date(row.get("datetime", ""))
        if pd.isna(article_date):
            article_date = latest_date
        days = max((latest_date - article_date).days, 0)
        time_decay = math.exp(-days / 14.0)

        for company in companies:
            node_stats[company]["article_count"] += 1
            if is_negative:
                node_stats[company]["negative_article_count"] += 1

        for source, target in combinations(sorted(set(companies)), 2):
            key = tuple(sorted((source, target)))
            edge_stats[key]["co_count"] += 1
            edge_stats[key]["time_decay_sum"] += time_decay
            if is_negative:
                edge_stats[key]["negative_co_count"] += 1
            if title and len(edge_stats[key]["sample_titles"]) < 3:
                edge_stats[key]["sample_titles"].append(title)

    nodes = pd.DataFrame(
        [
            {
                "company": company,
                "article_count": values["article_count"],
                "negative_article_count": values["negative_article_count"],
            }
            for company, values in node_stats.items()
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "source": source,
                "target": target,
                "co_count": values["co_count"],
                "negative_co_count": values["negative_co_count"],
                "time_decay": values["time_decay_sum"] / max(values["co_count"], 1),
                "sample_titles": "；".join(values["sample_titles"]),
            }
            for (source, target), values in edge_stats.items()
        ]
    )

    if edges.empty:
        return nodes, edges

    max_co = max(float(edges["co_count"].max()), 1.0)
    max_neg = max(float(edges["negative_co_count"].max()), 1.0)
    edges["co_strength"] = (edges["co_count"] / max_co).clip(0, 1)
    edges["neg_strength"] = (edges["negative_co_count"] / max_neg).clip(0, 1)
    edges["edge_weight"] = (
        0.5 * edges["co_strength"]
        + 0.3 * edges["neg_strength"]
        + 0.2 * edges["time_decay"].clip(0, 1)
    ).round(4)
    return nodes, edges.sort_values("edge_weight", ascending=False).reset_index(drop=True)


def compute_risk_propagation(edges, risk_table, source_company, max_depth=2, top_n=10):
    if edges is None or edges.empty or risk_table is None or risk_table.empty or not source_company:
        return pd.DataFrame()

    latest_date = risk_table["date"].max()
    latest = risk_table[risk_table["date"] == latest_date].copy()
    risk_map = latest.set_index("company")["risk_score"].to_dict()
    source_risk = float(risk_map.get(source_company, 50.0))

    adjacency = defaultdict(list)
    for row in edges.itertuples(index=False):
        adjacency[row.source].append((row.target, float(row.edge_weight), int(row.co_count), int(row.negative_co_count), row.sample_titles))
        adjacency[row.target].append((row.source, float(row.edge_weight), int(row.co_count), int(row.negative_co_count), row.sample_titles))

    rows = []
    visited_best = {source_company: 1.0}
    queue = deque([(source_company, 0, 1.0, [source_company])])

    while queue:
        current, depth, path_prob, path = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor, edge_weight, co_count, neg_count, sample_titles in adjacency.get(current, []):
            if neighbor in path:
                continue
            propagation_prob = float(np.clip((source_risk / 100.0) * path_prob * edge_weight, 0, 1))
            next_depth = depth + 1
            if propagation_prob <= visited_best.get(neighbor, 0):
                continue
            visited_best[neighbor] = propagation_prob
            next_path = path + [neighbor]
            neighbor_risk = float(risk_map.get(neighbor, 0.0))
            rows.append(
                {
                    "source_company": source_company,
                    "affected_company": neighbor,
                    "propagation_probability": round(propagation_prob, 4),
                    "spread_strength": round(propagation_prob * 100, 2),
                    "depth": next_depth,
                    "path": " -> ".join(next_path),
                    "edge_weight": round(edge_weight, 4),
                    "co_count": co_count,
                    "negative_co_count": neg_count,
                    "target_current_risk": round(neighbor_risk, 2),
                    "explain": f"与风险源存在 {co_count} 次新闻共现，其中负面共现 {neg_count} 次；当前路径层级为 {next_depth}。",
                    "sample_titles": sample_titles,
                }
            )
            queue.append((neighbor, next_depth, propagation_prob, next_path))

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("propagation_probability", ascending=False).head(top_n).reset_index(drop=True)


def build_network_figure(edges, propagation, source_company):
    shown = {source_company}
    if propagation is not None and not propagation.empty:
        for path in propagation["path"]:
            shown.update([item.strip() for item in str(path).split("->")])

    graph_edges = []
    for row in edges.itertuples(index=False):
        if row.source in shown and row.target in shown:
            graph_edges.append(row)

    nodes = sorted(shown)
    if not nodes:
        return go.Figure()

    radius = 1.0
    positions = {}
    positions[source_company] = (0, 0)
    others = [n for n in nodes if n != source_company]
    for idx, node in enumerate(others):
        angle = 2 * math.pi * idx / max(len(others), 1)
        positions[node] = (radius * math.cos(angle), radius * math.sin(angle))

    fig = go.Figure()
    for edge in graph_edges:
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=max(1, edge.edge_weight * 5), color="rgba(56,189,248,0.45)"),
                hoverinfo="text",
                text=f"{edge.source} - {edge.target}<br>边权: {edge.edge_weight}<br>共现: {edge.co_count}",
                showlegend=False,
            )
        )

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    probabilities = {}
    if propagation is not None and not propagation.empty:
        probabilities = propagation.set_index("affected_company")["propagation_probability"].to_dict()

    for node in nodes:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        prob = 1.0 if node == source_company else float(probabilities.get(node, 0.1))
        node_text.append(f"{node}<br>传播概率: {prob:.2%}" if node != source_company else f"{node}<br>风险源")
        node_size.append(34 if node == source_company else 18 + prob * 30)
        node_color.append("#ef4444" if node == source_company else "#38bdf8")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="#ffffff")),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=470,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(color="#ffffff"),
    )
    return fig


def render_graph_risk_module(st, df_news, risk_table, entity_label="企业", default_source=""):
    st.markdown("---")
    st.subheader("关联风险传导图谱")
    st.caption("基于企业共现关系图，模拟负面舆情在关联主体之间的扩散路径，辅助判断风险会影响谁、沿什么路径扩散。")

    if not os.path.exists("news_with_companies.csv"):
        st.warning("未找到 news_with_companies.csv，无法构建企业关系图谱。")
        return

    news_graph = pd.read_csv("news_with_companies.csv")
    nodes, edges = build_company_graph(news_graph, df_news)
    if edges.empty:
        st.warning("当前新闻数据未形成可用的企业共现边。")
        return

    latest_date = risk_table["date"].max()
    latest = risk_table[risk_table["date"] == latest_date].copy()
    graph_companies = set(edges["source"]).union(set(edges["target"]))
    high_sources = latest[(latest["company"].isin(graph_companies)) & (latest["risk_score"] >= 50)]
    if high_sources.empty:
        high_sources = latest[latest["company"].isin(graph_companies)]

    options = high_sources.sort_values("risk_score", ascending=False)["company"].drop_duplicates().tolist()
    if default_source in graph_companies and default_source not in options:
        options.insert(0, default_source)
    if not options:
        st.warning("未找到同时具备风险分和图谱关系的企业。")
        return

    default_index = options.index(default_source) if default_source in options else 0
    source_company = st.selectbox(f"选择风险源{entity_label}：", options, index=default_index)
    propagation = compute_risk_propagation(edges, risk_table, source_company, max_depth=2, top_n=5)

    c1, c2, c3 = st.columns(3)
    c1.metric("图谱节点数", f"{len(nodes)}")
    c2.metric("图谱边数", f"{len(edges)}")
    c3.metric("潜在受影响对象", f"{len(propagation)}")

    if propagation.empty:
        st.info("该风险源暂未识别出明显的二层内传播路径。")
        return

    left, right = st.columns([1.15, 1])
    with left:
        fig = build_network_figure(edges, propagation, source_company)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### 受影响对象 Top5")
        display = propagation[
            [
                "affected_company",
                "propagation_probability",
                "spread_strength",
                "depth",
                "path",
                "co_count",
                "negative_co_count",
                "explain",
            ]
        ].copy()
        display["propagation_probability"] = (display["propagation_probability"] * 100).round(2).astype(str) + "%"
        st.dataframe(
            display.rename(
                columns={
                    "affected_company": f"潜在受影响{entity_label}",
                    "propagation_probability": "传播概率",
                    "spread_strength": "扩散强度",
                    "depth": "传播层级",
                    "path": "传播路径",
                    "co_count": "共现次数",
                    "negative_co_count": "负面共现",
                    "explain": "解释原因",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

    top = propagation.iloc[0]
    st.info(
        f"图谱解释：{source_company} 的风险最可能沿路径 `{top['path']}` 传播，"
        f"对 {top['affected_company']} 的传播概率约为 {top['propagation_probability']:.2%}。"
        "该结果由共现强度、负面共现和时间衰减共同计算得到。"
    )

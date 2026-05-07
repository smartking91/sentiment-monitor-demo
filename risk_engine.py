import numpy as np
import pandas as pd

try:
    from graph_risk_engine import render_graph_risk_module
    GRAPH_RISK_AVAILABLE = True
except ImportError:
    GRAPH_RISK_AVAILABLE = False

LEVEL_ORDER = ["高风险", "中风险", "关注", "低风险"]
LEVEL_COLORS = {
    "高风险": "#ef4444",
    "中风险": "#f59e0b",
    "关注": "#38bdf8",
    "低风险": "#10b981",
}


def _contains_any(series, keywords):
    result = pd.Series(False, index=series.index)
    text = series.fillna("").astype(str)
    for keyword in keywords:
        result = result | text.str.contains(keyword, na=False, regex=False)
    return result


def _risk_level(score):
    if score >= 70:
        return "高风险"
    if score >= 50:
        return "中风险"
    if score >= 30:
        return "关注"
    return "低风险"


def _action_advice(level, entity_label):
    if level == "高风险":
        return (
            f"建议立即进入{entity_label}舆情应急响应：核查负面新闻来源，梳理核心事实，"
            "准备对外回应口径，并持续跟踪后续传播。"
        )
    if level == "中风险":
        return (
            f"建议将该{entity_label}纳入重点观察名单：提高监控频率，复核负面样本，"
            "评估是否需要主动沟通或发布补充说明。"
        )
    if level == "关注":
        return (
            f"建议保持常规监控：关注指数变化和新增负面内容，若连续下滑则升级预警。"
        )
    return "当前风险较低，建议维持日常监控并保留历史趋势作为基线。"


def normalize_news(df_news):
    news = df_news.copy()
    if news.empty:
        return news

    if "publish_time" in news.columns:
        news["date"] = news["publish_time"].astype(str).str[:10]
    elif "datetime" in news.columns:
        news["date"] = news["datetime"].astype(str).str[:10]
    elif "date" not in news.columns:
        news["date"] = ""

    if "sentiment" in news.columns:
        news["is_negative"] = _contains_any(news["sentiment"], ["负面", "利空"])
    else:
        news["is_negative"] = False

    for col in ["company", "title", "sentence_text", "url"]:
        if col not in news.columns:
            news[col] = ""
        news[col] = news[col].fillna("").astype(str)

    return news


def build_risk_table(df_scores, df_news=None):
    """Build daily company risk scores from aggregated score data and sentence news."""
    if df_scores is None or df_scores.empty:
        return pd.DataFrame()

    scores = df_scores.copy()
    required = ["date", "company", "news_count", "pos_count", "neg_count", "public_opinion_index"]
    missing = [col for col in required if col not in scores.columns]
    if missing:
        raise ValueError(f"company_daily_scores.csv 缺少字段: {', '.join(missing)}")

    scores["date"] = scores["date"].astype(str).str[:10]
    scores["company"] = scores["company"].fillna("").astype(str)
    for col in ["news_count", "pos_count", "neg_count", "public_opinion_index"]:
        scores[col] = pd.to_numeric(scores[col], errors="coerce").fillna(0)

    scores = scores[scores["company"].str.len() > 0].copy()
    scores = scores.sort_values(["company", "date"])

    scores["negative_rate"] = np.where(
        scores["news_count"] > 0,
        scores["neg_count"] / scores["news_count"],
        0,
    )
    scores["heat_rank"] = scores.groupby("date")["news_count"].rank(method="min", ascending=False)
    daily_max_heat = scores.groupby("date")["news_count"].transform("max").replace(0, np.nan)
    scores["heat_score"] = (scores["news_count"] / daily_max_heat * 100).fillna(0).clip(0, 100)

    scores["prev_index"] = scores.groupby("company")["public_opinion_index"].shift(1)
    scores["drop_1d"] = (scores["prev_index"] - scores["public_opinion_index"]).clip(lower=0).fillna(0)
    scores["index_3d_ago"] = scores.groupby("company")["public_opinion_index"].shift(3)
    scores["drop_3d"] = (scores["index_3d_ago"] - scores["public_opinion_index"]).clip(lower=0).fillna(0)
    scores["volatility_score"] = (scores["drop_1d"] * 1.5 + scores["drop_3d"]).clip(0, 100)

    diff = scores.groupby("company")["public_opinion_index"].diff()
    scores["decline_flag"] = diff < 0
    scores["decline_3d"] = (
        scores.groupby("company")["decline_flag"]
        .rolling(3, min_periods=3)
        .sum()
        .reset_index(level=0, drop=True)
        .eq(3)
    )

    scores["risk_score"] = (
        0.45 * (100 - scores["public_opinion_index"])
        + 0.25 * scores["negative_rate"] * 100
        + 0.20 * scores["volatility_score"]
        + 0.10 * scores["heat_score"]
    ).round(2).clip(0, 100)
    scores["risk_level"] = scores["risk_score"].apply(_risk_level)

    reason_rows = []
    for row in scores.itertuples(index=False):
        reasons = []
        if row.public_opinion_index < 40:
            reasons.append("舆情指数低于40")
        if row.negative_rate > 0.30:
            reasons.append("负面占比超过30%")
        if row.heat_rank <= 10:
            reasons.append("新闻热度进入Top 10")
        if bool(row.decline_3d):
            reasons.append("近3日指数连续下降")
        if row.neg_count >= 2:
            reasons.append("单日负面新闻数大于等于2")
        if not reasons:
            reasons.append("未触发强预警规则，按综合风险分监控")
        reason_rows.append("；".join(reasons))

    scores["trigger_reasons"] = reason_rows
    return scores.sort_values(["date", "risk_score"], ascending=[False, False]).reset_index(drop=True)


def get_latest_risks(risk_table):
    if risk_table is None or risk_table.empty:
        return pd.DataFrame(), ""
    latest_date = risk_table["date"].max()
    latest = risk_table[risk_table["date"] == latest_date].copy()
    return latest.sort_values("risk_score", ascending=False), latest_date


def get_related_news(df_news, company, date, limit=12):
    news = normalize_news(df_news)
    if news.empty:
        return news
    mask = (news["company"] == company) & (news["date"] == str(date)[:10])
    related = news[mask].copy()
    if related.empty:
        return related
    related["negative_sort"] = related["is_negative"].astype(int)
    return related.sort_values("negative_sort", ascending=False).head(limit)


def build_backtest_summary(risk_table):
    if risk_table is None or risk_table.empty:
        return pd.DataFrame()
    summary = (
        risk_table.groupby(["date", "risk_level"])
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )
    return summary


def render_risk_warning_center(st, px, df_news, df_scores, entity_label="企业"):
    risk_table = build_risk_table(df_scores, df_news)
    latest, latest_date = get_latest_risks(risk_table)

    st.title(f"风控预警中心")
    st.markdown(
        f"基于舆情指数、负面占比、热度异常和连续下跌信号，对{entity_label}进行可解释风险预警。"
    )

    if risk_table.empty or latest.empty:
        st.warning("暂无可用于风险预警的数据。")
        return

    high_count = int((latest["risk_level"] == "高风险").sum())
    medium_count = int((latest["risk_level"] == "中风险").sum())
    watch_count = int((latest["risk_level"] == "关注").sum())
    avg_score = float(latest["risk_score"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("预警日期", latest_date)
    c2.metric("高风险对象", f"{high_count} 个")
    c3.metric("中风险对象", f"{medium_count} 个")
    c4.metric("平均风险分", f"{avg_score:.1f}")

    st.markdown("---")
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader(f"历史风险{entity_label}清单")
        display_cols = [
            "date",
            "company",
            "risk_level",
            "risk_score",
            "public_opinion_index",
            "negative_rate",
            "news_count",
            "neg_count",
            "trigger_reasons",
        ]
        top_risks = latest[display_cols].head(20).copy()
        top_risks["negative_rate"] = (top_risks["negative_rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            top_risks.rename(
                columns={
                    "date": "日期",
                    "company": entity_label,
                    "risk_level": "预警等级",
                    "risk_score": "风险分",
                    "public_opinion_index": "舆情指数",
                    "negative_rate": "负面占比",
                    "news_count": "新闻数",
                    "neg_count": "负面数",
                    "trigger_reasons": "触发原因",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=430,
        )

    with right:
        st.subheader("风险等级概览")
        level_counts = latest["risk_level"].value_counts().reindex(LEVEL_ORDER, fill_value=0).reset_index()
        level_counts.columns = ["risk_level", "count"]
        fig_level = px.bar(
            level_counts,
            x="risk_level",
            y="count",
            color="risk_level",
            color_discrete_map=LEVEL_COLORS,
            template="plotly_dark",
            text="count",
        )
        fig_level.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=False,
            xaxis_title="预警等级",
            yaxis_title=f"{entity_label}数量",
        )
        st.plotly_chart(fig_level, use_container_width=True)
        st.info(
            f"当前共有 {high_count} 个高风险、{medium_count} 个中风险、"
            f"{watch_count} 个关注对象。"
        )

    st.markdown("---")
    st.subheader(f"{entity_label}风险画像")

    company_options = latest["company"].tolist()
    default_index = 0
    selected_company = st.selectbox(f"选择{entity_label}查看近7日风险画像：", company_options, index=default_index)
    company_history = (
        risk_table[risk_table["company"] == selected_company]
        .sort_values("date")
        .tail(7)
        .copy()
    )

    h1, h2 = st.columns([1.25, 1])
    with h1:
        fig_trend = px.line(
            company_history,
            x="date",
            y=["public_opinion_index", "risk_score"],
            markers=True,
            template="plotly_dark",
            labels={"date": "日期", "value": "分值", "variable": "指标"},
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            xaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
            yaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)", range=[0, 100]),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with h2:
        latest_company = company_history.iloc[-1]
        st.metric("当前风险分", f"{latest_company['risk_score']:.1f}")
        st.metric("当前舆情指数", f"{latest_company['public_opinion_index']:.1f}")
        st.metric("当前负面占比", f"{latest_company['negative_rate'] * 100:.1f}%")
        st.warning(f"触发原因：{latest_company['trigger_reasons']}")
        st.success(_action_advice(latest_company["risk_level"], entity_label))

    st.markdown("---")
    st.subheader("风险证据切片")
    related = get_related_news(df_news, selected_company, latest_company["date"])
    if related.empty:
        st.caption("该日期暂无可回溯的句子级新闻明细。")
    else:
        news_cols = ["date", "title", "sentiment", "sentence_text", "url"]
        available_cols = [col for col in news_cols if col in related.columns]
        st.dataframe(
            related[available_cols].rename(
                columns={
                    "date": "日期",
                    "title": "标题",
                    "sentiment": "情感",
                    "sentence_text": "关联句子",
                    "url": "链接",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

    if GRAPH_RISK_AVAILABLE:
        render_graph_risk_module(
            st,
            df_news,
            risk_table,
            entity_label=entity_label,
            default_source=selected_company,
        )
    else:
        st.warning("未找到 graph_risk_engine.py，无法加载关系图谱传播模块。")

    st.markdown("---")
    st.subheader("历史预警模拟")
    backtest = build_backtest_summary(risk_table)
    fig_backtest = px.bar(
        backtest,
        x="date",
        y="count",
        color="risk_level",
        color_discrete_map=LEVEL_COLORS,
        template="plotly_dark",
        labels={"date": "日期", "count": f"{entity_label}数量", "risk_level": "预警等级"},
    )
    fig_backtest.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
        yaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

    case_cols = ["date", "company", "risk_level", "risk_score", "trigger_reasons"]
    st.dataframe(
        risk_table[case_cols].head(12).rename(
            columns={
                "date": "日期",
                "company": entity_label,
                "risk_level": "预警等级",
                "risk_score": "风险分",
                "trigger_reasons": "触发原因",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _response_window(level):
    if level == "高风险":
        return "24小时内"
    if level == "中风险":
        return "3天内"
    if level == "关注":
        return "7天内"
    return "常规巡检"


def render_risk_warning_center(st, px, df_news, df_scores, entity_label="企业"):
    risk_table = build_risk_table(df_scores, df_news)
    latest, latest_date = get_latest_risks(risk_table)

    st.title("企业经营风险决策台")
    st.markdown("主线：**风险源识别 -> 触发原因解释 -> 关联风险传导 -> 分阶段处置方案**")

    if risk_table.empty or latest.empty:
        st.warning("暂无可用于风险预警的评分数据。")
        return

    latest = latest.sort_values("risk_score", ascending=False).copy()
    high_count = int((latest["risk_level"] == "高风险").sum())
    avg_score = float(latest["risk_score"].mean())
    default_company = latest.iloc[0]["company"]

    source_options = latest["company"].drop_duplicates().tolist()
    selected_company = st.selectbox(
        "选择重点风险源企业：",
        source_options,
        index=source_options.index(default_company) if default_company in source_options else 0,
    )

    company_history = (
        risk_table[risk_table["company"] == selected_company]
        .sort_values("date")
        .tail(7)
        .copy()
    )
    latest_company = company_history.iloc[-1]
    response_window = _response_window(latest_company["risk_level"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("当前预警等级", latest_company["risk_level"])
    m2.metric("高风险对象数", f"{high_count} 个")
    m3.metric("平均风险分", f"{avg_score:.1f}")
    m4.metric("处置时效提示", response_window)

    st.markdown("---")
    st.subheader(f"{selected_company} 风险画像")
    left, right = st.columns([1.3, 1])

    with left:
        fig_trend = px.line(
            company_history,
            x="date",
            y=["risk_score", "public_opinion_index"],
            markers=True,
            template="plotly_dark",
            labels={"date": "日期", "value": "分值", "variable": "指标"},
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            xaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
            yaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)", range=[0, 100]),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with right:
        st.metric("风险分", f"{latest_company['risk_score']:.1f}")
        st.metric("舆情指数", f"{latest_company['public_opinion_index']:.1f}")
        st.metric("负面占比", f"{latest_company['negative_rate'] * 100:.1f}%")
        st.metric("当日负面声量", f"{int(latest_company['neg_count'])} 条")
        st.warning(f"触发原因：{latest_company['trigger_reasons']}")

    st.markdown("---")
    st.subheader("三阶段处置建议")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error(
            "**24小时：事实核查与口径统一**\n\n"
            "1. 锁定风险源新闻与核心质疑点。\n"
            "2. 核查关联方、供应链、价格或声誉异常。\n"
            "3. 输出统一回应口径和内部责任人。"
        )
    with c2:
        st.warning(
            "**3-7天：重点修复与外部沟通**\n\n"
            "1. 对高传播节点进行定向澄清。\n"
            "2. 补充经营进展、合规说明或服务承诺。\n"
            "3. 跟踪竞对动态与舆情扩散变化。"
        )
    with c3:
        st.success(
            "**14-30天：复盘沉淀与机制优化**\n\n"
            "1. 复盘评分触发因子和处置时效。\n"
            "2. 更新风险词库与重点监测名单。\n"
            "3. 形成企业经营风险月度基线。"
        )

    if GRAPH_RISK_AVAILABLE:
        render_graph_risk_module(
            st,
            df_news,
            risk_table,
            entity_label=entity_label,
            default_source=selected_company,
        )
    else:
        st.warning("未找到 graph_risk_engine.py，无法加载关联风险传导图谱。")

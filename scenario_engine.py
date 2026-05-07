# scenario_engine.py
# 适配 final_company_sentiment.csv:
# publish_time / company / sentiment / sentence_text
# 电商四大场景真实评分计算 + 对接前端可视化

import os
import pandas as pd
import numpy as np


# ===================== 电商四大场景配置 =====================

SCENARIO_KEYWORDS = {
    "竞对监控": {
        "pos_words": [
            "增长", "提升", "突破", "领先", "市场份额", "热销", "爆款",
            "销量增长", "订单增长", "用户增长", "价格优势", "渠道扩张",
            "新品发布", "品牌升级", "合作", "中标", "获批"
        ],
        "neg_words": [
            "降价", "价格战", "竞争加剧", "份额下滑", "销量下滑",
            "毛利率承压", "库存高企", "竞品冲击", "渠道收缩",
            "投诉", "差评", "退货", "亏损"
        ],
        "weights": {
            "价格竞争力": 0.30,
            "活动策略": 0.25,
            "口碑对比": 0.25,
            "市场份额": 0.20
        },
        "dim_keywords": {
            "价格竞争力": ["价格", "降价", "提价", "毛利率", "成本", "补贴", "价格战"],
            "活动策略": ["促销", "活动", "直播", "618", "双11", "新品", "渠道", "营销"],
            "口碑对比": ["口碑", "投诉", "差评", "好评", "用户", "消费者", "满意度"],
            "市场份额": ["市场份额", "销量", "订单", "领先", "增长", "竞品", "份额"]
        }
    },

    "风险传导": {
        "pos_words": [
            "供应链稳定", "产能恢复", "成本下降", "政策支持", "需求回暖",
            "订单充足", "交付改善", "行业景气", "合作稳定", "恢复增长"
        ],
        "neg_words": [
            "供应链中断", "原材料涨价", "库存压力", "需求下滑", "政策收紧",
            "行业下行", "产能过剩", "交付延迟", "停产", "减产",
            "风险传导", "连带风险"
        ],
        "weights": {
            "供应链稳定": 0.35,
            "政策影响": 0.25,
            "竞品牵连": 0.20,
            "行业景气": 0.20
        },
        "dim_keywords": {
            "供应链稳定": ["供应链", "原材料", "交付", "产能", "库存", "停产", "复产"],
            "政策影响": ["政策", "监管", "补贴", "关税", "合规", "处罚", "审批"],
            "竞品牵连": ["竞品", "同业", "同行", "龙头", "上下游", "产业链", "传导"],
            "行业景气": ["行业", "景气", "需求", "周期", "下行", "回暖", "扩张"]
        }
    },

    "品牌声誉": {
        "pos_words": [
            "品牌提升", "口碑良好", "用户认可", "好评", "增长", "创新",
            "社会责任", "服务升级", "产品升级", "质量提升", "市场认可"
        ],
        "neg_words": [
            "投诉", "差评", "质量问题", "虚假宣传", "售后问题", "退货",
            "召回", "舆论质疑", "用户不满", "负面评价", "品牌受损",
            "信任危机"
        ],
        "weights": {
            "用户口碑": 0.35,
            "差评控制": 0.30,
            "社媒声量": 0.20,
            "品牌好感": 0.15
        },
        "dim_keywords": {
            "用户口碑": ["口碑", "用户", "消费者", "评价", "认可", "满意"],
            "差评控制": ["差评", "投诉", "退货", "质量问题", "售后", "召回"],
            "社媒声量": ["热搜", "舆论", "社媒", "媒体", "关注", "发酵"],
            "品牌好感": ["品牌", "形象", "好感", "信任", "声誉", "社会责任"]
        }
    },

    "平台合规": {
        "pos_words": [
            "合规经营", "规范运营", "平台治理", "整改完成", "审核通过",
            "知识产权保护", "内容安全", "风控升级", "管理规范",
            "广告合规"
        ],
        "neg_words": [
            "违规", "处罚", "下架", "封禁", "侵权", "假冒", "虚假宣传",
            "广告违规", "内容违规", "知识产权纠纷", "监管问询",
            "行政处罚", "立案调查"
        ],
        "weights": {
            "广告合规": 0.30,
            "知识产权": 0.25,
            "内容安全": 0.25,
            "平台规则": 0.20
        },
        "dim_keywords": {
            "广告合规": ["广告", "虚假宣传", "营销", "宣传", "违规宣传", "夸大"],
            "知识产权": ["知识产权", "侵权", "专利", "商标", "版权", "假冒"],
            "内容安全": ["内容", "下架", "封禁", "审核", "违规内容", "安全"],
            "平台规则": ["平台", "规则", "处罚", "合规", "监管", "整改", "问询"]
        }
    }
}


# ===================== 加载舆情 CSV 数据 =====================

FINANCIAL_ENTERPRISE_SCENARIOS = {
    "经营风险预警": {
        "pos_words": [
            "融资", "授信", "中标", "回款", "订单", "增长", "盈利", "现金流改善", "偿债能力",
            "供应链稳定", "政策支持", "合作", "扩产", "增持", "评级上调"
        ],
        "neg_words": [
            "违约", "逾期", "抽贷", "诉讼", "被执行", "失信", "债务", "流动性", "亏损",
            "供应链中断", "停产", "监管", "处罚", "问询", "关联方", "担保", "风险传染"
        ],
        "weights": {
            "关联方风险": 0.40,
            "偿债能力": 0.25,
            "供应链稳定": 0.20,
            "行业政策": 0.15
        },
        "dim_keywords": {
            "关联方风险": ["关联方", "供应商", "客户", "担保", "股东", "子公司", "投资方", "合作方", "风险传染", "诉讼"],
            "偿债能力": ["债务", "违约", "逾期", "偿债", "现金流", "融资", "授信", "债券", "兑付", "抽贷"],
            "供应链稳定": ["供应链", "停产", "交付", "订单", "库存", "原材料", "采购", "生产", "供货", "物流"],
            "行业政策": ["政策", "监管", "处罚", "问询", "行业", "合规", "税收", "补贴", "标准", "许可"]
        }
    },
    "品牌声誉监测": {
        "pos_words": [
            "澄清", "回应", "道歉", "整改", "口碑", "认可", "获奖", "公益", "服务升级",
            "品牌升级", "正面评价", "用户增长"
        ],
        "neg_words": [
            "谣言", "投诉", "差评", "争议", "曝光", "舆论", "造假", "质量问题", "虚假宣传",
            "抵制", "道歉", "负面传播", "关联账号", "水军", "热搜"
        ],
        "weights": {
            "传播异常": 0.35,
            "信源可信": 0.25,
            "情绪突变": 0.20,
            "关联账号": 0.20
        },
        "dim_keywords": {
            "传播异常": ["热搜", "转发", "传播", "扩散", "爆料", "曝光", "短视频", "社交平台", "舆论", "发酵"],
            "信源可信": ["央视", "新华社", "证券时报", "公告", "监管", "法院", "官方", "媒体", "财报", "交易所"],
            "情绪突变": ["愤怒", "质疑", "抵制", "差评", "投诉", "不满", "道歉", "澄清", "争议", "信任"],
            "关联账号": ["账号", "大V", "博主", "自媒体", "水军", "关联", "矩阵", "评论", "转发", "粉丝"]
        }
    },
    "竞对动态洞察": {
        "pos_words": [
            "市场份额", "提价", "新品", "中标", "扩张", "合作", "订单增长", "渠道", "优势",
            "竞品", "招投标", "工商变更"
        ],
        "neg_words": [
            "降价", "价格战", "竞品", "替代", "份额下滑", "招投标失败", "工商变更", "减持",
            "撤回", "亏损", "大促", "补贴", "渠道流失"
        ],
        "weights": {
            "价格变动": 0.30,
            "工商变更": 0.25,
            "招投标": 0.25,
            "舆情情绪": 0.20
        },
        "dim_keywords": {
            "价格变动": ["价格", "降价", "提价", "价格战", "折扣", "补贴", "促销", "大促", "报价", "毛利"],
            "工商变更": ["工商", "股权", "法人", "注册资本", "变更", "减持", "增持", "并购", "投资", "控股"],
            "招投标": ["招标", "投标", "中标", "废标", "项目", "采购", "合同", "订单", "竞标", "标的"],
            "舆情情绪": ["负面", "正面", "争议", "热度", "关注", "评价", "投诉", "认可", "舆情", "口碑"]
        }
    }
}

SCENARIO_KEYWORDS.clear()
SCENARIO_KEYWORDS.update(FINANCIAL_ENTERPRISE_SCENARIOS)


def load_news_data():
    csv_path = "final_company_sentiment.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "publish_time" in df.columns:
            df["publish_time"] = df["publish_time"].astype(str).str[:10]

        for col in ["company", "sentiment", "sentence_text", "title"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        return df

    return pd.DataFrame()


df_news_global = load_news_data()


# ===================== 工具函数 =====================

def safe_contains(series, keyword):
    return series.astype(str).str.contains(keyword, na=False, regex=False)


def get_text_score_by_keywords(text, pos_words, neg_words):
    pos_hits = [w for w in pos_words if w in text]
    neg_hits = [w for w in neg_words if w in text]
    return pos_hits, neg_hits


def calc_dimension_score(dim, dim_keywords, target_news, pos_ratio, neg_ratio):
    """
    每个维度独立算分：
    基础分 65；
    正面比例加分；
    负面比例扣分；
    维度关键词命中时，根据该维度下相关句子的情感再微调。
    """
    base = 65 + pos_ratio * 25 - neg_ratio * 30

    if target_news.empty:
        return 65

    keywords = dim_keywords.get(dim, [])

    if not keywords:
        return round(max(35, min(100, base)), 0)

    text_series = target_news["sentence_text"].fillna("").astype(str)

    dim_mask = pd.Series(False, index=target_news.index)

    for kw in keywords:
        dim_mask = dim_mask | text_series.str.contains(kw, na=False, regex=False)

    dim_news = target_news[dim_mask]

    if dim_news.empty:
        # 没有维度关键词，稍微保守一点，但仍受总体情绪影响
        score = base - 3
    else:
        dim_total = len(dim_news)
        dim_pos = dim_news["sentiment"].str.contains("正面", na=False).sum()
        dim_neg = dim_news["sentiment"].str.contains("负面", na=False).sum()

        dim_pos_ratio = dim_pos / dim_total if dim_total else 0
        dim_neg_ratio = dim_neg / dim_total if dim_total else 0

        score = 62 + dim_pos_ratio * 35 - dim_neg_ratio * 40

    return round(max(35, min(100, score)), 0)


def build_advice_text(company_name, scenario_type, grade_code):
    if scenario_type == "经营风险预警":
        return f"""
### 风险提示
1. 重点关注{company_name}的关联方风险、偿债能力、供应链稳定和行业政策变化。
2. 若新闻中集中出现诉讼、债务、抽贷、供应链中断等信号，需警惕融资或投标受阻。
3. 关联主体的负面事件可能传导至目标企业，形成隐性经营风险。

### 应对建议
1. 建立供应商、客户、股东和担保方的关联风险清单。
2. 对高风险关联方进行补充尽调，提前准备替代供应或信用说明材料。
3. 在融资、授信、投标等关键窗口期提高预警等级，形成处置闭环。
"""

    if scenario_type == "品牌声誉监测":
        return f"""
### 风险提示
1. 重点关注{company_name}的传播异常、信源可信度、情绪突变和关联账号扩散。
2. 若负面信息由高可信媒体或多账号矩阵扩散，可能迅速形成声誉压力。
3. 舆论发酵初期若未及时回应，容易造成“谣言未热，澄清已晚”的被动局面。

### 应对建议
1. 对高传播度负面内容进行信源分级和传播链路追踪。
2. 准备澄清说明、问答口径和客服响应模板，缩短处置时间。
3. 联动正面案例、权威背书和服务改进信息，修复品牌信任。
"""

    if scenario_type == "竞对动态洞察":
        return f"""
### 风险提示
1. 重点关注{company_name}所在市场的价格变动、工商变更、招投标和舆情情绪。
2. 若竞品出现降价、渠道扩张、中标或工商资本动作，可能影响市场份额。
3. 竞对异动常常先于销售结果出现，应在大促、投标、招商前提前监控。

### 应对建议
1. 建立竞品价格、招投标和工商变更的联动监测机制。
2. 对竞品降价或中标事件生成差异化定价、组合优惠和渠道维护建议。
3. 将竞对舆情热度与销售、库存、毛利指标结合，及时调整经营策略。
"""

    if scenario_type == "竞对监控":
        return f"""
### 风险提示
1. 关注{company_name}与同类品牌/竞品的价格、促销和渠道策略变化。
2. 若负面声量集中在销量、份额或毛利率，需要警惕竞争优势削弱。
3. 持续跟踪竞品新品发布、价格战和市场份额转移。

### 应对建议
1. 建立竞品价格与促销活动监测机制。
2. 将舆情热度、正负面比例与销售数据联动分析。
3. 对高热度负面事件设置预警阈值，及时调整营销策略。
"""

    if scenario_type == "风险传导":
        return f"""
### 风险提示
1. 关注{company_name}上游原材料、供应链、政策和行业周期变化。
2. 若新闻集中提及库存、交付、停产、需求下滑，需警惕风险传导。
3. 行业负面事件可能通过产业链扩散至目标主体。

### 应对建议
1. 建立供应链和行业政策舆情监控看板。
2. 对上下游高频负面关键词设置自动预警。
3. 将舆情风险与经营、库存、订单指标结合判断。
"""

    if scenario_type == "品牌声誉":
        return f"""
### 风险提示
1. 关注{company_name}是否出现投诉、差评、质量问题或舆论质疑。
2. 若负面内容在短时间集中出现，可能造成品牌声誉扩散风险。
3. 用户口碑变化会影响转化率、复购率和品牌信任。

### 应对建议
1. 建立差评和投诉快速响应机制。
2. 对高传播度负面新闻进行原因归类和处置跟踪。
3. 通过正面案例、服务升级和产品改进修复品牌信任。
"""

    return f"""
### 风险提示
1. 关注{company_name}是否涉及广告违规、知识产权纠纷、平台处罚等合规风险。
2. 若出现下架、处罚、侵权、虚假宣传等信号，应及时排查业务流程。
3. 平台规则变化可能影响内容投放、商品展示和经营稳定性。

### 应对建议
1. 建立平台规则与合规关键词监控机制。
2. 对广告宣传、知识产权和内容安全进行定期自查。
3. 将合规舆情预警接入运营审核和法务响应流程。
"""


# ===================== 核心评分函数 =====================

def get_scenario_analysis(company_name, date_str, scenario_type):
    base_score = 65
    final_score = 65
    score_change = 0
    grade_code = "L2(中风险)"
    grade_desc = "舆情状态稳定，需保持常规监测"
    dimension_scores = {}
    add_list = []
    sub_list = []
    desc = "基于真实新闻数据量化计算"
    advice_text = ""

    # 关键：只保留企业经营三大场景，默认回到 C-RATE 经营风险预警
    if scenario_type not in SCENARIO_KEYWORDS:
        scenario_type = "经营风险预警"

    cfg = SCENARIO_KEYWORDS[scenario_type]
    pos_words = cfg["pos_words"]
    neg_words = cfg["neg_words"]
    weights = cfg["weights"]
    dim_keywords = cfg["dim_keywords"]

    if not df_news_global.empty:
        company_mask = df_news_global["company"].astype(str).str.contains(
            str(company_name), na=False, regex=False
        )

        if date_str and date_str in df_news_global["publish_time"].values:
            date_mask = df_news_global["publish_time"] == date_str
        else:
            latest_date = df_news_global["publish_time"].max()
            date_mask = df_news_global["publish_time"] == latest_date
            date_str = latest_date

        target_news = df_news_global[company_mask & date_mask].copy()

        if not target_news.empty:
            total_news = len(target_news)
            pos_count = target_news["sentiment"].str.contains("正面", na=False).sum()
            neg_count = target_news["sentiment"].str.contains("负面", na=False).sum()

            pos_ratio = pos_count / total_news if total_news > 0 else 0
            neg_ratio = neg_count / total_news if total_news > 0 else 0

            all_text = " ".join(target_news["sentence_text"].fillna("").astype(str).tolist())
            pos_hits, neg_hits = get_text_score_by_keywords(all_text, pos_words, neg_words)

            for word in pos_hits[:5]:
                add_list.append(f"新闻提及【{word}】，对【{scenario_type}】形成正向信号")

            if not add_list:
                add_list.append(f"正面声量占比 {pos_ratio * 100:.1f}%，整体情绪相对稳定")

            for word in neg_hits[:5]:
                sub_list.append(f"新闻提及【{word}】，对【{scenario_type}】存在潜在风险")

            if not sub_list:
                sub_list.append(f"负面声量占比 {neg_ratio * 100:.1f}%，暂未发现集中风险信号")

            # 每个场景自己的维度独立评分
            for dim in weights.keys():
                dimension_scores[dim] = calc_dimension_score(
                    dim=dim,
                    dim_keywords=dim_keywords,
                    target_news=target_news,
                    pos_ratio=pos_ratio,
                    neg_ratio=neg_ratio
                )

            final_score = 0
            for dim, w in weights.items():
                final_score += dimension_scores[dim] * w

            final_score = round(final_score, 1)
            score_change = round(final_score - base_score, 1)

            if final_score >= 80:
                grade_code = "L1(低风险)"
                grade_desc = "舆情整体向好，暂未发现明显扩散风险"
            elif 60 <= final_score < 80:
                grade_code = "L2(中风险)"
                grade_desc = "存在一定波动信号，建议保持常规监测"
            else:
                grade_code = "L3(高风险)"
                grade_desc = "负面舆情集中，建议立即介入处置"

            desc = (
                f"本次分析【{company_name}】在 {date_str} 共 {total_news} 条细粒度舆情记录，"
                f"正面占比 {pos_ratio * 100:.1f}%，负面占比 {neg_ratio * 100:.1f}%，"
                f"按【{scenario_type}】专属维度进行加权评分。"
            )

    advice_text = build_advice_text(company_name, scenario_type, grade_code)

    return {
        "final_score": final_score,
        "score_change": score_change,
        "grade_code": grade_code,
        "grade_desc": grade_desc,
        "dimension_scores": dimension_scores,
        "basis": {
            "加分事件": add_list,
            "扣分事件": sub_list,
            "维度说明": desc
        },
        "advice_text": advice_text
    }

#!/usr/bin/env python3
"""
explore_neo4j.py — 连接 Neo4j 数据库，导出完整 Schema 并保存到 schema_output.txt。

用法:
    export NEO4J_PASSWORD="your_password"
    python explore_neo4j.py

环境变量（所有敏感信息必须通过环境变量传入，不设默认值）:
    NEO4J_URL      : Neo4j Bolt URL（例 bolt://100.104.181.96:7687）
    NEO4J_USER     : Neo4j 用户名（默认 neo4j）
    NEO4J_PASSWORD : Neo4j 密码（必填，无默认值）
    NEO4J_DATABASE : 数据库名（默认 neo4j）

输出:
    schema_output.txt — 完整的 Schema 分析报告（Markdown 格式）
"""

import os
import sys
import json
from datetime import datetime
from typing import Any

# ============================================================
# 配置区 —— 所有敏感信息仅从环境变量读取
# ============================================================
NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# 脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "schema_output.txt")


# ============================================================
# Neo4j 连接
# ============================================================
def get_driver():
    """创建并返回 Neo4j driver。"""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[ERROR] 未安装 neo4j 驱动。请运行: pip install neo4j")
        sys.exit(1)

    if not NEO4J_PASSWORD:
        print("[ERROR] 未设置 NEO4J_PASSWORD 环境变量。")
        print("  用法: export NEO4J_PASSWORD='your_password' && python explore_neo4j.py")
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver


# ============================================================
# 查询函数
# ============================================================
def run_cypher(driver, cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
    """执行 Cypher 查询，返回 dict 列表。"""
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


def query_node_counts(driver) -> list[dict]:
    print("[INFO] 查询各标签节点数量 ...")
    return run_cypher(driver, """
        MATCH (n) UNWIND labels(n) AS label
        RETURN label, count(*) AS count ORDER BY count DESC
    """)


def query_node_labels_and_props(driver) -> list[dict]:
    print("[INFO] 查询所有节点标签及属性 ...")
    return run_cypher(driver, """
        MATCH (n) RETURN DISTINCT labels(n) AS labels, keys(n) AS properties LIMIT 200
    """)


def query_relationship_types(driver) -> list[dict]:
    print("[INFO] 查询所有关系类型 ...")
    return run_cypher(driver, """
        MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship_type ORDER BY relationship_type
    """)


def query_relationship_counts(driver) -> list[dict]:
    print("[INFO] 查询各关系类型数量 ...")
    return run_cypher(driver, """
        MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count ORDER BY count DESC
    """)


def query_relationship_keys(driver) -> list[dict]:
    print("[INFO] 查询各关系类型属性键 ...")
    return run_cypher(driver, """
        MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship_type, keys(r) AS property_keys ORDER BY relationship_type
    """)


def query_node_examples(driver) -> list[dict]:
    print("[INFO] 查询各节点标签属性详情（含示例值）...")
    return run_cypher(driver, """
        MATCH (n) WITH labels(n) AS labels, n UNWIND labels AS label
        WITH label, n LIMIT 500
        RETURN label, collect(DISTINCT keys(n))[0] AS property_keys,
               properties(n) AS example_properties LIMIT 50
    """)


def query_indexes(driver) -> list[dict]:
    print("[INFO] 查询索引 ...")
    return run_cypher(driver, "SHOW INDEXES")


def query_constraints(driver) -> list[dict]:
    print("[INFO] 查询约束 ...")
    return run_cypher(driver, "SHOW CONSTRAINTS")


def query_schema_viz(driver) -> list[dict]:
    print("[INFO] 调用 db.schema.visualization() ...")
    return run_cypher(driver, "CALL db.schema.visualization()")


def query_patterns(driver) -> list[dict]:
    print("[INFO] 查询图谱连接模式 ...")
    return run_cypher(driver, """
        MATCH (a)-[r]->(b) RETURN DISTINCT labels(a) AS source_labels,
        type(r) AS relationship, labels(b) AS target_labels ORDER BY relationship LIMIT 50
    """)


# ============================================================
# 格式化输出
# ============================================================
def fmt_section(title: str) -> str:
    return f"\n\n{'='*70}\n  {title}\n{'='*70}\n"


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ============================================================
# 主逻辑
# ============================================================
def main() -> None:
    print("=" * 60)
    print("  Neo4j Schema 探索脚本")
    print(f"  地址: {NEO4J_URL}  数据库: {NEO4J_DATABASE}")
    print("=" * 60)

    driver = get_driver()

    try:
        driver.verify_connectivity()
        print("[OK] Neo4j 连接成功！\n")

        out: list[str] = []
        out.append(f"# Neo4j Schema 探索报告\n\n")
        out.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        out.append(f"数据库: `{NEO4J_DATABASE}` | 地址: `{NEO4J_URL}`\n\n")

        # 1. db.schema.visualization
        out.append(fmt_section("1. 图谱结构概览 (db.schema.visualization)"))
        try:
            sv = query_schema_viz(driver)
            out.append(f"```json\n{json.dumps(sv, indent=2, ensure_ascii=False, default=str)}\n```\n")
        except Exception as e:
            out.append(f"失败: {e}\n")

        # 2. 节点数量
        out.append(fmt_section("2. 节点标签及数量"))
        nc = query_node_counts(driver)
        out.append(fmt_table(["Label", "Count"], [[r.get("label","?"), str(r.get("count",0))] for r in nc]))

        # 3. 节点属性键
        out.append(fmt_section("3. 节点标签及属性键"))
        nl = query_node_labels_and_props(driver)
        lp: dict[str, set] = {}
        for item in nl:
            for lbl in item.get("labels", []):
                lp.setdefault(lbl, set()).update(item.get("properties", []))
        for lbl in sorted(lp):
            out.append(f"\n### `{lbl}`\n属性: `{', '.join(sorted(lp[lbl]))}`\n")

        # 4. 节点示例
        out.append(fmt_section("4. 节点属性详情（含示例值）"))
        for item in query_node_examples(driver):
            out.append(f"\n### `{item.get('label','?')}` 示例\n")
            out.append(f"```json\n{json.dumps(item.get('example_properties',{}), indent=2, ensure_ascii=False, default=str)}\n```\n")

        # 5. 关系数量
        out.append(fmt_section("5. 关系类型及数量"))
        rc = query_relationship_counts(driver)
        out.append(fmt_table(["Relationship", "Count"], [[r.get("relationship_type","?"), str(r.get("count",0))] for r in rc]))

        # 6. 关系属性
        out.append(fmt_section("6. 关系类型及属性键"))
        for item in query_relationship_keys(driver):
            out.append(f"- **`{item.get('relationship_type','?')}`**: `{', '.join(sorted(item.get('property_keys',[])))}`\n")

        # 7. 索引
        out.append(fmt_section("7. 索引"))
        try:
            idx = query_indexes(driver)
            if idx:
                keys = list(idx[0].keys())
                out.append(fmt_table(keys, [[str(r.get(k,"")) for k in keys] for r in idx]))
            else:
                out.append("(无索引)\n")
        except Exception as e:
            out.append(f"查询失败: {e}\n")

        # 8. 约束
        out.append(fmt_section("8. 约束"))
        try:
            cst = query_constraints(driver)
            if cst:
                keys = list(cst[0].keys())
                out.append(fmt_table(keys, [[str(r.get(k,"")) for k in keys] for r in cst]))
            else:
                out.append("(无约束)\n")
        except Exception as e:
            out.append(f"查询失败: {e}\n")

        # 9. 连接模式
        out.append(fmt_section("9. 图谱连接模式"))
        for p in query_patterns(driver):
            out.append(f"`(:{', '.join(p.get('source_labels',['?']))}) -[:{p.get('relationship','?')}]-> (:{', '.join(p.get('target_labels',['?']))})`\n")

        # 写入文件
        full = "".join(out)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(full)
        print(f"\n[OK] 报告已保存到: {OUTPUT_FILE} ({len(full):,} 字节)")
        print(full)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        driver.close()
        print("\n[DONE] 连接已关闭。")


if __name__ == "__main__":
    main()

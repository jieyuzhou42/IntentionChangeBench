"""把一轮的候选池压成结构化证据，交给 judge。

为什么需要这个:

WebShop 那条链路早就在做同样的事 —— run_human_annotated_pilot_eval._action_evidence
在 domain == 'webshop' 时会按 selected_asin 去 candidate_items 里查出完整记录，
连同 action 一起给 judge，所以 judge 判"这个商品满不满足约束"时手上有商品的全部字段。

TravelPlanner 这条链路 `return {"action": action}` 就结束了，从来不附候选表。后果是
judge 只能核对 agent 主动写出来的属性：实测 10 条 instance 的住宿行里只有 1 条提到了
minimum stay，于是 16/53 轮 agent 订了最低入住晚数根本满足不了的房源，judge 在其中
8 轮判成 satisfied。这不是 judge 判错，是证据里压根没有那个字段。

WebShop 靠 ASIN 精确 lookup，TravelPlanner 的 agent 写的是自由文本的酒店名
（"Sunny 2 bd + Private rooftop on best GP street, cost $293"），没法精确匹配，
所以锚定这一步交给 judge 自己做 —— 模糊名称匹配正是 LLM 的强项，用 regex 去解析
agent 的自由文本才是死路。这里只负责把池子整理成它能查的样子。
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List

# 只保留会被约束用到的字段。整池原样塞进 prompt 会把经纬度、电话、网址一起带进去，
# 纯属烧 token。
ACCOMMODATION_FIELDS = (
    "NAME", "price", "room type", "minimum nights",
    "maximum occupancy", "review rate number", "house_rules",
)
RESTAURANT_FIELDS = ("Name", "Average Cost", "Cuisines", "Aggregate Rating")
ATTRACTION_FIELDS = ("Name", "Address")
FLIGHT_FIELDS = (
    "Flight Number", "Price", "DepTime", "ArrTime", "ActualElapsedTime",
    "FlightDate", "OriginCityName", "DestCityName",
)


def _compact(item: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    out = {}
    for field in fields:
        value = item.get(field)
        if value is not None and value != "":
            out[field] = copy.deepcopy(value)
    return out


def _pool_items(turn: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    results = (turn.get("env_feedback") or {}).get("search_results") or {}
    for page in results.get(key) or []:
        for item in page.get("items") or []:
            if isinstance(item, dict):
                out.append(item)
    return out


def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def candidate_pool(instance: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """整条 instance 的候选池并集。

    按 instance 而不是按轮取: 同一条 instance 各轮的搜索结果基本相同，按轮传会把
    几乎一样的内容重复 7 次。并集里多出来的选项不会造成误判 —— judge 要做的是
    "agent 写的这个对应哪一条"，池子大一点只是多几个候选。
    """
    accommodations: List[Dict[str, Any]] = []
    restaurants: List[Dict[str, Any]] = []
    attractions: List[Dict[str, Any]] = []
    transportation: List[Dict[str, Any]] = []

    for turn in instance.get("turns") or []:
        for item in _pool_items(turn, "accommodations"):
            accommodations.append(_compact(item, ACCOMMODATION_FIELDS))
        for item in _pool_items(turn, "restaurants"):
            restaurants.append(_compact(item, RESTAURANT_FIELDS))
        for item in _pool_items(turn, "attractions"):
            attractions.append(_compact(item, ATTRACTION_FIELDS))
        for item in _pool_items(turn, "transportation"):
            if item.get("Flight Number"):
                transportation.append(_compact(item, FLIGHT_FIELDS))
            elif item.get("value"):
                # 自驾/出租车/"没有航班"都是这种散文条目，原样保留。
                transportation.append({"value": str(item["value"])})

    return {
        "accommodations": _dedup(accommodations),
        "restaurants": _dedup(restaurants),
        "attractions": _dedup(attractions),
        "transportation": _dedup(transportation),
    }


def pool_size_note(pool: Dict[str, List[Dict[str, Any]]]) -> str:
    return "  ".join(f"{key}={len(value)}" for key, value in pool.items())

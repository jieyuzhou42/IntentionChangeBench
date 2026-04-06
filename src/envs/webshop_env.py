from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import AgentAction, EnvFeedback
from envs.base_env import BaseEnv
import re


class WebShopEnvAdapter(BaseEnv):
    """
    Adapter for the real WebShop text environment.

    Supported usage pattern:
        import gym
        from web_agent_site.envs import WebAgentTextEnv
        env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)

    This adapter tries to be robust to:
    - old/new gym step signatures
    - different observation representations
    - action string variants such as click[...] vs choose[...]
    """

    def __init__(self, webshop_env, action_style: str = "auto"):
        """
        Args:
            webshop_env: actual gym env instance created by gym.make(...)
            action_style:
                - "auto": try click[...] first, then choose[...]
                - "click"
                - "choose"
        """
        self.webshop_env = webshop_env
        self.action_style = action_style
        self.last_raw_observation: Any = None
        self.last_observation: Dict[str, Any] = {}
        self.last_info: Dict[str, Any] = {}
        self.done = False

    def reset(self, task=None) -> Dict[str, Any]:
        """
        Real WebShop env usually samples a random instruction on reset.
        We do not rely on task.initial_intention here, because the public README
        documents reset() but does not document a stable public API for directly
        setting a custom instruction.
        """
        reset_out = self.webshop_env.reset()

        if isinstance(reset_out, tuple):
            raw_obs = reset_out[0]
            info = reset_out[1] if len(reset_out) > 1 else {}
        else:
            raw_obs = reset_out
            info = {}

        self.last_raw_observation = raw_obs
        self.last_info = info or {}
        self.done = False

        obs = self._normalize_observation(raw_obs, self.last_info)
        self.last_observation = obs
        return obs

    def get_observation(self) -> Dict[str, Any]:
        return self.last_observation

    def step(self, agent_action: AgentAction, user_state: Dict[str, Any]) -> EnvFeedback:
        candidate_actions = self._serialize_action_candidates(agent_action)

        last_error = None
        step_out = None
        used_action = None

        for action_text in candidate_actions:
            try:
                step_out = self.webshop_env.step(action_text)
                used_action = action_text
                last_error = None
                break
            except Exception as e:
                last_error = e

        if step_out is None:
            return EnvFeedback(
                status="error",
                feasible=False,
                reason=f"env_action_error: {repr(last_error)}",
                observation=self.last_observation,
                result={},
                satisfied_constraints=[],
                violated_constraints=self._get_active_constraint_fields(user_state),
            )

        raw_obs, reward, done, info = self._unpack_step_output(step_out)
        self.last_raw_observation = raw_obs
        self.last_info = info or {}
        self.done = done

        obs = self._normalize_observation(raw_obs, self.last_info)
        self.last_observation = obs

        result = self._extract_result(obs, info=self.last_info)
        satisfied, violated = self._check_constraints(result, user_state)
        active_constraints = self._get_active_constraint_fields(user_state)

        if self._looks_like_no_results(obs):
            return EnvFeedback(
                status="observed",
                feasible=True,
                reason="no_matching_results",
                observation={**obs, "executed_action": used_action},
                result=result,
                satisfied_constraints=satisfied,
                violated_constraints=violated or active_constraints,
            )

        # reward can optionally help detect terminal purchase success
        if violated and satisfied:
            return EnvFeedback(
                status="observed",
                feasible=True,
                reason="partial_constraint_failure",
                observation={**obs, "executed_action": used_action, "reward": reward},
                result=result,
                satisfied_constraints=satisfied,
                violated_constraints=violated,
            )

        if violated and not satisfied:
            return EnvFeedback(
                status="observed",
                feasible=True,
                reason="constraint_mismatch",
                observation={**obs, "executed_action": used_action, "reward": reward},
                result=result,
                satisfied_constraints=satisfied,
                violated_constraints=violated,
            )

        return EnvFeedback(
            status="observed",
            feasible=True,
            reason=None,
            observation={**obs, "executed_action": used_action, "reward": reward},
            result=result,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
        )

    def get_instruction_text(self) -> str:
        """
        Publicly observed in WebShop issue discussions.
        Falls back to parsing the last observation if the helper is unavailable.
        """
        if hasattr(self.webshop_env, "get_instruction_text"):
            try:
                return self.webshop_env.get_instruction_text()
            except Exception:
                pass

        return self.last_observation.get("instruction", "")

    def get_available_actions(self) -> Dict[str, Any]:
        """
        Try to read available actions from env helper if present, else parse from observation.
        README examples show available actions in text env output.
        """
        return self._safe_get_available_actions(self.last_observation)

    def _safe_get_available_actions(self, fallback_obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if hasattr(self.webshop_env, "get_available_actions"):
            try:
                actions = self.webshop_env.get_available_actions()
                if isinstance(actions, dict):
                    return actions
            except Exception:
                pass

        return self._infer_available_actions_from_obs(fallback_obs or self.last_observation)

    def parse_instruction_to_intention(self, instruction: str) -> Optional[dict]:
        if not instruction or not instruction.strip():
            return None

        cleaned_instruction = instruction.replace("Instruction:", "").strip()
        if not cleaned_instruction:
            return None

        text = cleaned_instruction.lower()

        constraints = {
            "category": None,
            "color": None,
            "budget_max": None,
            "brand": None,
        }

        m = re.search(r"price lower than ([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            constraints["budget_max"] = float(m.group(1))

        color_markers = ["black", "white", "green", "blue", "red", "pink", "grey", "gray"]
        for c in color_markers:
            if c in text:
                constraints["color"] = c
                break

        # very rough category extraction
        if "chair" in text:
            constraints["category"] = "office chair"
        elif "jumpsuits" in text or "rompers" in text or "overalls" in text:
            constraints["category"] = "jumpsuit"

        return {
            "request": cleaned_instruction,
            "constraints": constraints,
            "priority": ["category", "budget_max", "color", "brand"],
        }

    def _unpack_step_output(self, step_out):
        """
        Supports both old gym (obs, reward, done, info)
        and newer gymnasium (obs, reward, terminated, truncated, info).
        """
        if isinstance(step_out, tuple):
            if len(step_out) == 4:
                raw_obs, reward, done, info = step_out
                return raw_obs, reward, done, info
            if len(step_out) == 5:
                raw_obs, reward, terminated, truncated, info = step_out
                return raw_obs, reward, bool(terminated or truncated), info

        # fallback
        return step_out, 0.0, False, {}

    def _serialize_action_candidates(self, agent_action: AgentAction) -> List[str]:
        at = agent_action.action_type
        payload = agent_action.action_payload or {}

        if at == "search":
            query = payload.get("query", "")
            return [f"search[{query}]"]

        if at == "click":
            target = payload.get("target", "")
            if self.action_style == "click":
                return [f"click[{target}]"]
            if self.action_style == "choose":
                return [f"choose[{target}]"]
            return [f"click[{target}]", f"choose[{target}]"]

        if at == "buy":
            if self.action_style == "click":
                return ["click[Buy Now]", "click[Buy]"]
            if self.action_style == "choose":
                return ["choose[Buy Now]", "choose[Buy]"]
            return [
                "click[Buy Now]",
                "click[Buy]",
                "choose[Buy Now]",
                "choose[Buy]",
            ]

        if at == "back_to_search":
            if self.action_style == "click":
                return ["click[Back to Search]"]
            if self.action_style == "choose":
                return ["choose[Back to Search]"]
            return ["click[Back to Search]", "choose[Back to Search]"]

        if at == "next_page":
            if self.action_style == "click":
                return ["click[Next >]", "click[Next Page]"]
            if self.action_style == "choose":
                return ["choose[Next >]", "choose[Next Page]"]
            return [
                "click[Next >]",
                "click[Next Page]",
                "choose[Next >]",
                "choose[Next Page]",
            ]

        if at == "prev_page":
            if self.action_style == "click":
                return ["click[< Prev]", "click[Prev Page]"]
            if self.action_style == "choose":
                return ["choose[< Prev]", "choose[Prev Page]"]
            return [
                "click[< Prev]",
                "click[Prev Page]",
                "choose[< Prev]",
                "choose[Prev Page]",
            ]

        if at == "refine":
            query = payload.get("query", "")
            return [f"search[{query}]"]

        return ["search[]"]

    def _normalize_observation(self, raw_obs: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        WebShop text env observations are often text-heavy.
        We normalize into a dict while preserving raw text.
        """
        info = info or {}

        obs_text = raw_obs if isinstance(raw_obs, str) else str(raw_obs)
        instruction = ""
        if hasattr(self.webshop_env, "get_instruction_text"):
            try:
                instruction = self.webshop_env.get_instruction_text()
            except Exception:
                instruction = ""
        if not instruction:
            instruction = self._extract_instruction_from_text(obs_text)

        page_type = self._infer_page_type(obs_text, info)
        available_actions = self._safe_get_available_actions(self.last_observation)
        clickables = self._extract_clickables(info, obs_text, available_actions)
        visible_items = self._extract_visible_items_from_text(obs_text, clickables, page_type)
        selected_asin = self._current_selected_asin()
        selected_options = self._current_selected_options()
        item_context = self._extract_item_context(page_type, selected_asin, selected_options)

        return {
            "page_type": page_type,
            "instruction": instruction,
            "raw_text": obs_text,
            "clickables": clickables,
            "visible_items": visible_items,
            "selected_item": self._extract_selected_item(obs_text, page_type, item_context=item_context),
            "selected_asin": selected_asin,
            "selected_options": selected_options,
            "item_context": item_context,
            "info": info,
        }

    def _infer_page_type(self, obs_text: str, info: Dict[str, Any]) -> str:
        t = obs_text.lower()

        if "buy now" in t or "product details" in t or "item page" in t:
            return "item"
        if "results" in t or "page 1" in t or "back to search" in t:
            return "results"
        if "instruction:" in t and "search" in t:
            return "search"

        # fallback from clickables
        clickables = info.get("clickables", [])
        if "search" in clickables:
            return "search"

        return "unknown"

    def _extract_instruction_from_text(self, text: str) -> str:
        # README examples show text env observations include "Instruction:"
        marker = "Instruction:"
        if marker in text:
            after = text.split(marker, 1)[1].strip()
            # stop early if another known section begins
            for stop in ["[SEP]", "Available actions:", "Search", "Results"]:
                idx = after.find(stop)
                if idx > 0:
                    after = after[:idx].strip()
            return after
        return ""

    def _extract_clickables(
        self,
        info: Dict[str, Any],
        obs_text: str,
        available_actions: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        clickables = info.get("clickables")
        if isinstance(clickables, list):
            return clickables

        if isinstance(available_actions, dict):
            clickables = available_actions.get("clickables")
            if isinstance(clickables, list):
                return clickables

        # weak text fallback
        return []

    def _extract_visible_items_from_text(
        self,
        raw_text: str,
        clickables: Optional[List[str]] = None,
        page_type: str = "unknown",
    ) -> List[Dict[str, Any]]:
        if page_type != "results":
            return []

        parts = [p.strip() for p in raw_text.split("[SEP]")]
        items = []
        i = 0
        while i < len(parts) - 2:
            asin = parts[i]
            title = parts[i + 1]
            price_text = parts[i + 2]

            if self._looks_like_asin(asin) and "$" in price_text:
                price = self._parse_price(price_text)
                items.append({
                    "asin": asin,
                    "title": title,
                    "price": price,
                    "click_target": asin.lower() if clickables and asin.lower() in clickables else asin,
                })
                i += 3
            else:
                i += 1
        return items

    def _extract_selected_item(
        self,
        obs_text: str,
        page_type: str,
        item_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if page_type != "item":
            return None
        parts = [p.strip() for p in obs_text.split("[SEP]") if p.strip()]
        title = None
        price = None
        for idx, part in enumerate(parts):
            if part.lower().startswith("price:"):
                price = self._parse_price(part)
                if idx > 0:
                    title = parts[idx - 1]
                break

        if title is None:
            title = self._guess_title_from_item_page(obs_text)

        if item_context:
            title = title or item_context.get("title")
            if price is None:
                pricing = item_context.get("pricing") or []
                if pricing:
                    price = pricing[0]
                else:
                    price = item_context.get("price")

        if title is None and price is None:
            return None

        selected = {"title": title, "price": price}
        if item_context:
            for key in ("asin", "category", "product_category", "rating", "brand", "color"):
                if item_context.get(key) is not None:
                    selected[key] = item_context[key]
        return selected

    def _guess_title_from_item_page(self, obs_text: str) -> Optional[str]:
        lines = [x.strip() for x in obs_text.splitlines() if x.strip()]
        if not lines:
            return None
        # heuristic only
        for line in lines[:10]:
            if "instruction:" not in line.lower() and len(line) < 200:
                return line
        return None

    def _looks_like_no_results(self, obs: Dict[str, Any]) -> bool:
        text = obs.get("raw_text", "").lower()

        if obs.get("visible_items"):
            return False

        total_match = re.search(r"total results:\s*(\d+)", text)
        if total_match:
            return int(total_match.group(1)) == 0

        if "no results" in text or "did not match any products" in text:
            return True

        return False

    def _extract_result(self, obs: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        First real-env version:
        - result is lightweight unless you add custom parsing from observation text
        - price/color/brand/category are parsed heuristically when possible
        """
        info = info or {}
        item_context = obs.get("item_context") or {}
        selected = obs.get("selected_item")
        if selected:
            result = dict(selected)
            product_text = result.get("title", "") or ""
        elif obs.get("visible_items"):
            result = dict(obs["visible_items"][0])
            product_text = result.get("title", "") or ""
        else:
            result = {}
            product_text = obs.get("raw_text", "")

        if result.get("price") is not None:
            product_text = f"{product_text} ${result['price']}".strip()

        parsed_attrs = self._parse_product_attrs_from_text(product_text)
        result.update(parsed_attrs)
        if item_context:
            enriched_result = {
                "asin": item_context.get("asin"),
                "title": item_context.get("title"),
                "price": item_context.get("price"),
                "pricing": item_context.get("pricing"),
                "rating": item_context.get("rating"),
                "category": item_context.get("category"),
                "product_category": item_context.get("product_category"),
                "query": item_context.get("query"),
                "brand": item_context.get("brand"),
                "color": item_context.get("color"),
                "selected_options": item_context.get("selected_options"),
            }
            result.update({k: v for k, v in enriched_result.items() if v is not None})
        return result

    def _extract_item_context(
        self,
        page_type: str,
        selected_asin: Optional[str],
        selected_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if page_type != "item" or not selected_asin:
            return None

        product = self._lookup_product(selected_asin)
        if not isinstance(product, dict):
            return None

        pricing = product.get("pricing") or []
        price = pricing[0] if pricing else self._parse_price(str(product.get("Price", "")))
        bullet_points = product.get("BulletPoints") or []
        if not isinstance(bullet_points, list):
            bullet_points = [str(bullet_points)]
        reviews = product.get("Reviews") or []
        normalized_reviews: List[Dict[str, Any]] = []
        if isinstance(reviews, list):
            for review in reviews[:3]:
                if not isinstance(review, dict):
                    continue
                normalized_reviews.append(
                    {
                        "score": review.get("score"),
                        "summary": review.get("summary"),
                        "body": review.get("body"),
                    }
                )

        return {
            "asin": selected_asin,
            "title": product.get("Title") or product.get("name"),
            "price": price,
            "pricing": pricing[:2],
            "category": product.get("category"),
            "product_category": product.get("product_category"),
            "query": product.get("query"),
            "description": product.get("Description"),
            "bullet_points": bullet_points[:8],
            "rating": product.get("Rating"),
            "attributes": (product.get("Attributes") or [])[:12],
            "main_image": product.get("MainImage"),
            "options": product.get("options") or {},
            "selected_options": dict(selected_options or {}),
            "option_to_image": product.get("option_to_image") or {},
            "reviews": normalized_reviews,
            "instruction_text": product.get("instruction_text"),
            "instruction_attributes": product.get("instruction_attributes"),
            "brand": self._infer_brand_from_product(product),
            "color": self._infer_color_from_product(product),
        }

    def _lookup_product(self, asin: str) -> Optional[Dict[str, Any]]:
        server = getattr(self.webshop_env, "server", None)
        product_item_dict = getattr(server, "product_item_dict", None)
        if isinstance(product_item_dict, dict):
            product = product_item_dict.get(asin)
            if isinstance(product, dict):
                return product
        return None

    def _get_session_state(self) -> Dict[str, Any]:
        server = getattr(self.webshop_env, "server", None)
        session_id = getattr(self.webshop_env, "session", None)
        user_sessions = getattr(server, "user_sessions", None)
        if session_id is None or not isinstance(user_sessions, dict):
            return {}
        session = user_sessions.get(session_id)
        return session if isinstance(session, dict) else {}

    def _current_selected_asin(self) -> Optional[str]:
        session = self._get_session_state()
        asin = session.get("asin")
        return str(asin).strip().upper() if asin else None

    def _current_selected_options(self) -> Dict[str, Any]:
        session = self._get_session_state()
        options = session.get("options") or {}
        return dict(options) if isinstance(options, dict) else {}

    def _infer_brand_from_product(self, product: Dict[str, Any]) -> Optional[str]:
        for key in ("brand", "Brand"):
            value = product.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        title = str(product.get("Title", "") or "").strip()
        if not title:
            return None
        head = re.split(r"[-:|,]", title, maxsplit=1)[0].strip()
        if not head:
            return None
        words = head.split()
        if not words:
            return None
        return " ".join(words[:3])

    def _infer_color_from_product(self, product: Dict[str, Any]) -> Optional[str]:
        search_space = " ".join(
            [
                str(product.get("Title", "") or ""),
                str(product.get("Description", "") or ""),
                " ".join(str(x) for x in product.get("BulletPoints", []) or []),
            ]
        ).lower()
        for color in ["black", "white", "gray", "grey", "blue", "red", "green", "teal", "pink", "brown"]:
            if color in search_space:
                return color
        return None

    def _parse_product_attrs_from_text(self, text: str) -> Dict[str, Any]:
        """
        Heuristic parser only.
        Real WebShop text observations are not guaranteed to expose a stable structured schema.
        """
        out: Dict[str, Any] = {}

        lower = text.lower()

        # crude price parsing
        import re
        m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", text)
        if m:
            try:
                out["price"] = float(m.group(1))
            except ValueError:
                pass

        # crude category hints
        if "chair" in lower:
            out["category"] = "office chair"
        elif "jumpsuit" in lower or "jumpsuits" in lower or "rompers" in lower or "overalls" in lower:
            out["category"] = "jumpsuit"
        elif "shirt" in lower:
            out["category"] = "shirt"
        elif "shoe" in lower or "shoes" in lower:
            out["category"] = "shoes"

        # crude color hints
        for color in ["black", "white", "gray", "grey", "blue", "red", "green", "teal", "pink", "brown"]:
            if color in lower:
                out["color"] = color
                break

        return out

    def _infer_available_actions_from_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        clickables = obs.get("clickables", [])
        raw_text = obs.get("raw_text", "").lower()

        return {
            "has_search_bar": obs.get("page_type") == "search" or "search" in raw_text,
            "clickables": clickables,
        }

    def _get_active_constraint_fields(self, user_state: Dict[str, Any]) -> List[str]:
        constraints = user_state.get("constraints", {})
        return [field for field, value in constraints.items() if value is not None]

    def _looks_like_asin(self, text: str) -> bool:
        return bool(re.fullmatch(r"[A-Z0-9]{10}", text.strip()))

    def _parse_price(self, text: str) -> Optional[float]:
        m = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _check_constraints(self, result: Dict[str, Any], user_state: Dict[str, Any]) -> tuple[List[str], List[str]]:
        constraints = user_state.get("constraints", {})
        satisfied, violated = [], []

        if not result:
            # do not mark everything violated blindly if the env simply hasn't exposed the attrs yet
            return satisfied, []

        for field, desired in constraints.items():
            if desired is None:
                continue

            if field == "budget_max":
                price = result.get("price")
                if price is None:
                    continue
                if price <= desired:
                    satisfied.append(field)
                else:
                    violated.append(field)

            elif field in {"color", "brand", "category"}:
                actual = result.get(field)
                if actual is None:
                    continue
                if str(actual).lower() == str(desired).lower():
                    satisfied.append(field)
                else:
                    violated.append(field)

        return satisfied, violated

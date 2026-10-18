"""委派用的「模型角色」(model role) 解析。

為什麼需要這一層
----------------
``delegation.provider`` / ``delegation.model`` 是**全域**覆寫：一旦設定，
每一個子代理都換成那組模型。但常見需求是「只對某一個子任務換模型」——
例如重要變更完成後，主代理想請另一條路線的模型做一次獨立審查，而其他
子任務仍沿用原本的繼承行為。

本模組把 ``delegation.model_roles``（使用者寫在 config.yaml 的可信任設定）
映射成 provider / model / reasoning effort。設計上的硬性邊界：

* **只接受設定裡存在的角色名稱。**模型只能給一個角色字串，不能自己填
  endpoint、api_key，也不能猜模型 ID。
* **沒指定角色就完全不介入**，既有的全域覆寫與父模型繼承行為原封不動。
* **解析不出來就明確失敗。**絕不靜默換成別的模型後還宣稱完成了指定模型
  的審查。
* 角色名稱與既有的 ``role``（``leaf`` / ``orchestrator``）完全分離，保留字
  會被拒絕，避免兩個概念混用。

設定長相（``~/.hermes/config.yaml``）::

    delegation:
      model_roles:
        review:
          provider: openai
          model: <使用者自己填的模型 ID>
          reasoning_effort: high
        second_opinion:            # 選用的第二意見，不預設啟用
          provider: gemini
          model: <使用者自己填的模型 ID>

這裡刻意不內建任何「最新／最強」的模型 ID：模型 ID 一律由使用者設定，
本模組不追新版、也不假設跨供應商就比較強。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ModelRoleError",
    "MODEL_ROLES_CONFIG_KEY",
    "RESERVED_MODEL_ROLE_NAMES",
    "parse_model_roles",
    "available_model_role_names",
    "resolve_model_role",
    "resolved_role_identity",
]

#: ``delegation:`` 底下的設定鍵。
MODEL_ROLES_CONFIG_KEY = "model_roles"

#: 既有 delegation role 的保留字 —— 不可拿來當模型角色名，避免語意混用。
RESERVED_MODEL_ROLE_NAMES = frozenset({"leaf", "orchestrator"})

#: 角色定義裡認得的鍵。其餘鍵會被忽略並記 debug log（容忍使用者加註解式的鍵）。
_RECOGNISED_ROLE_KEYS = frozenset(
    {
        "provider",
        "model",
        "base_url",
        "api_key",
        "api_mode",
        "reasoning_effort",
        "description",
    }
)

#: 至少要有其中一個，否則這個角色沒有任何「選擇目標」可言。
_TARGET_KEYS = ("provider", "model", "base_url")

#: 角色名稱同時是工具 schema 的 enum 值，所以限制成安全、好讀的識別字。
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class ModelRoleError(ValueError):
    """指定的模型角色不存在、設定無效，或憑證解析失敗。

    呼叫端必須把這個錯誤原封不動回報給模型／使用者，不得改用其他模型續跑。
    """


def _normalise_name(value: Any) -> str:
    """把角色名稱正規化成小寫、去空白的形式。"""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _raw_role_block(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """取出 ``delegation.model_roles``；形狀不對就當作沒設定。"""
    if not isinstance(cfg, dict):
        return {}
    block = cfg.get(MODEL_ROLES_CONFIG_KEY)
    return block if isinstance(block, dict) else {}


def parse_model_roles(
    cfg: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """解析 ``delegation.model_roles``。

    Args:
        cfg: ``delegation`` 設定區塊（``_load_config()`` 的回傳值）。

    Returns:
        ``(valid, problems)``：

        * ``valid`` —— 角色名 → 正規化後的定義 dict，可直接餵給憑證解析器。
          另含 ``_reasoning_config`` 鍵（已解析好的 reasoning 設定，或 None）。
        * ``problems`` —— 角色名 → 人看得懂的原因字串。設定壞掉的角色不會
          進入 ``valid``（也就不會出現在工具 schema 裡），但原因保留下來，
          讓使用者指名這個角色時能收到真正的理由，而不是只有「不存在」。
    """
    from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort

    valid: Dict[str, Dict[str, Any]] = {}
    problems: Dict[str, str] = {}

    for raw_name, raw_defn in _raw_role_block(cfg).items():
        name = _normalise_name(raw_name)
        if not name:
            continue
        if name in RESERVED_MODEL_ROLE_NAMES:
            problems[name] = (
                f"'{name}' is reserved for the delegation role (leaf / "
                "orchestrator) and cannot be used as a model role name."
            )
            logger.warning("delegation.model_roles: %s", problems[name])
            continue
        if not _NAME_RE.match(name):
            problems[name] = (
                f"invalid role name '{raw_name}': use lowercase letters, "
                "digits, '_' or '-' (max 32 chars, must start with a letter)."
            )
            logger.warning("delegation.model_roles: %s", problems[name])
            continue
        if not isinstance(raw_defn, dict):
            problems[name] = (
                f"role '{name}' must be a mapping of provider/model/"
                f"reasoning_effort, got {type(raw_defn).__name__}."
            )
            logger.warning("delegation.model_roles: %s", problems[name])
            continue

        defn: Dict[str, Any] = {}
        for key, value in raw_defn.items():
            key_norm = str(key).strip().lower()
            if key_norm not in _RECOGNISED_ROLE_KEYS:
                logger.debug(
                    "delegation.model_roles.%s: ignoring unknown key %r", name, key
                )
                continue
            defn[key_norm] = value

        # 字串欄位去空白；空字串視為未設定，才不會把 "" 當成合法 provider。
        for key in ("provider", "model", "base_url", "api_key", "api_mode", "description"):
            if key in defn:
                defn[key] = str(defn[key] or "").strip() or None

        if not any(defn.get(key) for key in _TARGET_KEYS):
            problems[name] = (
                f"role '{name}' selects nothing: set at least one of "
                f"{', '.join(_TARGET_KEYS)}."
            )
            logger.warning("delegation.model_roles: %s", problems[name])
            continue

        # reasoning effort：不認得的值一律當設定錯誤處理。全域的
        # delegation.reasoning_effort 走「警告後繼承父層」的寬鬆路線，但這裡
        # 是使用者「指名」的審查角色，寬鬆退讓等於偷偷降級，寧可擋下來。
        reasoning_config = None
        if "reasoning_effort" in defn:
            raw_effort = defn["reasoning_effort"]
            parsed = parse_reasoning_effort(raw_effort)
            if parsed is None and raw_effort is not None and raw_effort is not True:
                problems[name] = (
                    f"role '{name}' has an unknown reasoning_effort "
                    f"'{raw_effort}'; valid values: none/false, "
                    f"{', '.join(VALID_REASONING_EFFORTS)}."
                )
                logger.warning("delegation.model_roles: %s", problems[name])
                continue
            reasoning_config = parsed
        defn["_reasoning_config"] = reasoning_config

        valid[name] = defn

    return valid, problems


def available_model_role_names(cfg: Optional[Dict[str, Any]]) -> List[str]:
    """回傳可用（設定正確）的角色名稱，排序固定。

    排序固定很重要：這份清單會進工具 schema 的 enum，順序飄動會讓
    system prompt／tool schema 不再 byte-stable，破壞 prompt cache。
    """
    valid, _problems = parse_model_roles(cfg)
    return sorted(valid)


def resolve_model_role(
    role_name: Any,
    cfg: Optional[Dict[str, Any]],
    parent_agent: Any,
    credential_resolver: Callable[[Dict[str, Any], Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """把一個角色名稱解析成子代理可用的憑證 bundle。

    Args:
        role_name: 模型／呼叫端指定的角色名。
        cfg: ``delegation`` 設定區塊。
        parent_agent: 父代理（供憑證解析器做繼承判斷）。
        credential_resolver: 與 ``_resolve_delegation_credentials`` 同簽名的
            函式 ``(cfg_like, parent_agent) -> creds dict``。由呼叫端注入，
            避免本模組反向依賴 ``tools.delegate_tool``。

    Returns:
        ``{"name", "credentials", "reasoning_config", "declared"}``。

    Raises:
        ModelRoleError: 角色不存在、設定無效或憑證解析失敗。永遠不回傳
            「退而求其次」的其他模型。
    """
    requested = _normalise_name(role_name)
    if not requested:
        raise ModelRoleError(
            "model_role must be a non-empty role name from "
            "delegation.model_roles in config.yaml."
        )

    valid, problems = parse_model_roles(cfg)

    if requested in problems:
        raise ModelRoleError(
            f"model_role '{requested}' is configured but unusable: "
            f"{problems[requested]} Fix delegation.model_roles in config.yaml, "
            "or omit model_role to keep the default delegation behavior. "
            "No substitute model was used."
        )

    if not valid:
        raise ModelRoleError(
            f"model_role '{requested}' was requested but no usable "
            "delegation.model_roles are configured in config.yaml. Configure "
            "the role first, or omit model_role to keep the default "
            "delegation behavior. No substitute model was used."
        )

    defn = valid.get(requested)
    if defn is None:
        raise ModelRoleError(
            f"Unknown model_role '{requested}'. Configured roles: "
            f"{', '.join(sorted(valid))}. Use one of those, or omit "
            "model_role to keep the default delegation behavior. "
            "No substitute model was used."
        )

    # 只把憑證相關的鍵交給解析器 —— 它吃的是 delegation 區塊的形狀。
    resolver_cfg = {
        key: defn[key]
        for key in ("provider", "model", "base_url", "api_key", "api_mode")
        if defn.get(key)
    }
    try:
        credentials = credential_resolver(resolver_cfg, parent_agent) or {}
    except Exception as exc:  # noqa: BLE001 —— 一律轉成明確的角色錯誤
        raise ModelRoleError(
            f"model_role '{requested}' "
            f"(provider={defn.get('provider') or 'inherited'}, "
            f"model={defn.get('model') or 'inherited'}) could not be "
            f"resolved: {exc} Fix delegation.model_roles in config.yaml. "
            "No substitute model was used."
        ) from exc

    credentials = dict(credentials)

    # 角色裡寫的 model 一定要贏：解析器在某些路徑會回傳 provider 的預設模型，
    # 那會讓「指名某個模型做審查」變成「跑了另一個模型」。
    if defn.get("model"):
        credentials["model"] = defn["model"]

    return {
        "name": requested,
        "credentials": credentials,
        "reasoning_config": defn.get("_reasoning_config"),
        "declared": {
            "provider": defn.get("provider"),
            "model": defn.get("model"),
            "reasoning_effort": defn.get("reasoning_effort"),
            "description": defn.get("description"),
        },
    }


def resolved_role_identity(resolved: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把 :func:`resolve_model_role` 的結果壓成可記錄的「解析後身分」。

    這是**設定推導出來的期望身分**，不是執行證據 —— 實際執行身分要從建好的
    child agent 物件讀（``child.model`` / ``child.provider``），兩者分開記錄，
    對不上就看得出來。
    """
    if not resolved:
        return None
    creds = resolved.get("credentials") or {}
    reasoning = resolved.get("reasoning_config")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            effort = "none"
        else:
            effort = reasoning.get("effort")
    else:
        effort = None
    return {
        "role": resolved.get("name"),
        "provider": creds.get("provider"),
        "model": creds.get("model"),
        "reasoning_effort": effort,
    }

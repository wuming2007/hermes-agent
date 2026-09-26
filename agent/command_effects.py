"""What a shell command line or an inline script would change, read without running it.

Heuristic and conservative on purpose. A command line is split into its simple commands (at
chains, pipes, newlines, a background ``&``, subshells and command substitutions; quoted text
stays inside its command, except where the shell itself substitutes); wrappers are peeled off
(``sudo``, ``env``, ``NAME=value``, ``timeout 60``, ``nohup``, ``xargs``, package runners such as
``npx``); nested scripts (``bash -c``, ``eval``, ``find -exec``) are read again; interpreter
one-liners (``python -c``, ``node -e``) are read by their file-writing and process-spawning calls.
A simple command that is none of the known reads, lookups, test runs or plain network checks has
an ``UNKNOWN`` effect.

The cognition evidence round and the working-memory scope gate both read commands here, so a
rewrite that hides a command from one hides it from the other too.
"""

from __future__ import annotations

import re
from typing import Callable

from agent.cognition_evidence import matching_command_kinds

AGENT_LAUNCH = "agent_launch"
MIGRATION = "migration"
REPO_CHANGE = "repo_change"
DEPLOY = "deploy"
FILE_WRITE = "file_write"
UNKNOWN = "unknown"
STATE_CHANGES = frozenset({AGENT_LAUNCH, MIGRATION, REPO_CHANGE, DEPLOY, FILE_WRITE})
_MAX_DEPTH = 4

# -- simple commands ------------------------------------------------------------------------------

# Command substitution runs inside double quotes too; the other separators only outside quotes.
_SUBSTITUTION = re.compile(r"\$\(|`")
_SEPARATOR = re.compile(r"\n|&&|\|\||\|&|\||;|(?<![<>&0-9])&(?![&>])|[()]")
_ASSIGNMENT = r"\w+=(?:\"[^\"]*\"|'[^']*'|\S*)"
# One wrapper at the start of a simple command; peeled until none is left. Every option token
# parses exactly one way (no two alternatives match the same text), so a long run of options
# cannot send the matcher into exponential backtracking.
_WRAPPER = re.compile(
    r"^(?:[{}!]"
    rf"|{_ASSIGNMENT}"
    r"|(?:sudo|doas)(?:\s+(?:-[ugCDhprtU]\s*\S+|-\S+))*"
    rf"|env(?:\s+(?:-[uSCP]\s*\S+|-\S+|{_ASSIGNMENT}))*"
    r"|(?:nohup|exec|builtin|time|caffeinate|stdbuf|ionice|chronic|unbuffer)(?:\s+-\S+)*"
    r"|command(?=\s+[^\s-])"
    r"|nice(?:\s+(?:-n\s*-?\d+|-\S+))*"
    r"|g?timeout(?:\s+(?:-[sk]\s+[^\s-]\S*|-[\w-]+(?:=\S+)?))*\s+\d\S*"
    r"|xargs(?:\s+(?:-[IiLnPdEs]\s*\S+|-\S+))*"
    r"|(?:npx|bunx|pnpx|uvx)(?:\s+(?:-p\s*\S+|--package[= ]\S+|-\S+))*"
    r"|(?:pnpm|yarn)\s+(?:dlx|exec)(?:\s+-\S+)*"
    r"|npm\s+exec(?:\s+-\S+)*"
    r"|(?:uv|poetry|pipenv|pdm|bundle)\s+(?:run|exec)(?:\s+-\S+)*"
    r"|pipx\s+run(?:\s+-\S+)*"
    r")(?:\s+|$)")
_QUOTED = {"'": re.compile(r"'([^']*)'"), '"': re.compile(r'"((?:[^"\\]|\\.)*)"')}


def _scan(command: str) -> list[tuple[str, str]]:
    """The simple commands of a command line, each with the operator that follows it ('' last)."""
    parts: list[tuple[str, str]] = []
    current: list[str] = []
    in_double = False
    i = 0
    while i < len(command):
        char = command[i]
        if char == "\\" and i + 1 < len(command):
            current.append(command[i:i + 2])
            i += 2
            continue
        if char == "'" and not in_double:
            end = command.find("'", i + 1)
            end = len(command) if end < 0 else end + 1
            current.append(command[i:end])
            i = end
            continue
        if char == '"':
            in_double = not in_double
        separator = _SUBSTITUTION.match(command, i) or (None if in_double else _SEPARATOR.match(command, i))
        if separator:
            parts.append(("".join(current).strip(), separator.group()))
            current = []
            i = separator.end()
            continue
        current.append(char)
        i += 1
    parts.append(("".join(current).strip(), ""))
    return parts


def _split(command: str) -> list[str]:
    """The simple commands of a command line."""
    return [step for step, _op in _scan(command) if step]


def _peel(step: str) -> str:
    """The command a wrapper runs: ``sudo env X=1 timeout 60 claude -p x`` runs ``claude -p x``."""
    for _ in range(16):
        peeled = _WRAPPER.sub("", step, count=1).lstrip("'\" \t")
        if peeled == step:
            break
        step = peeled
    return step


def _unquote(text: str) -> tuple[str, str]:
    """A quoted script and what follows it (``bash -c "x" > out``); an unclosed quote runs to the end."""
    text = text.strip()
    quoted = _QUOTED.get(text[:1])
    if quoted is None:
        return text, ""
    match = quoted.match(text)
    return (match[1], text[match.end():]) if match else (text[1:], "")


def _command_word(step: str) -> str:
    words = step.split(None, 1)
    return words[0].strip("'\"").rsplit("/", 1)[-1] if words else ""


# -- what one simple command changes --------------------------------------------------------------

_SHELL_C = re.compile(
    r"^(?:\S*/)?(?:ba|z|k|da|fi|c|tc|mk)?sh(?:\s+-[\w-]+)*?\s+-[a-zA-Z]*c[a-zA-Z]*\s+(?P<body>.+)", re.S)
_EVAL = re.compile(r"^eval\s+(?P<body>.+)", re.S)
_FIND_EXEC = re.compile(r"^(?:\S*/)?find\b.*?\s-(?:exec|execdir|ok|okdir)\s+(?P<body>.+)", re.S)
_IN_PLACE = re.compile(r"^(?:\S*/)?(?:perl|ruby)\b.*?\s-[a-zA-Z]*i", re.S)
_INLINE = re.compile(
    r"^(?:\S*/)?(?:python[\d.]*|pypy[\d.]*|node(?:js)?|bun|deno|ruby|perl|php|osascript|lua|Rscript)\b"
    r"(?:\s+-[\w-]+(?:=\S+)?)*?\s+(?:-[a-zA-Z]*[ceEr]|--eval|eval)\s+(?P<body>.+)", re.S)
_AGENT_CLI = (r"^(?:\S*/)?(?:claude(?:-code)?|codex|opencode(?:-ai)?|aider(?:-chat)?|gemini(?:-cli)?|cursor-agent)"
              r"(?![\w.-])|^(?:\S*/)?hermes\s+chat\b")
_AGENT_RUN = re.compile(_AGENT_CLI)  # an agent that works and reports back
_AGENT = re.compile(_AGENT_CLI + r"|^tmux\s+send-keys\b")  # also keys typed into a running agent
_MIGRATION = re.compile(
    r"^(?:\S*/)?(?:alembic|flyway|liquibase)\b|\bmanage\.py\s+migrate\b|^(?:\S*/)?prisma\s+migrate\b|\bdb:migrate\b")
_GIT = r"^(?:\S*/)?git(?:\s+(?:-[Cc]\s*\S+|--[\w-]+(?:=\S+)?))*\s+"
_GIT_PUSH = re.compile(_GIT + r"push\b")
_GIT_TREE = re.compile(  # rewrites files in the work tree
    _GIT + r"(?:checkout|switch|restore|reset|stash|pull|merge|rebase|cherry-pick|revert|am|apply|clean|rm|mv"
    r"|clone|init|worktree\s+(?:add|remove|move))\b")
_GIT_RECORD = re.compile(  # changes the repo's history, refs or settings
    _GIT + r"(?:commit|add|tag|fetch|notes|update-ref|filter-branch|gc|prune|submodule"
    r"|branch\s+(?:-[a-zA-Z]*[dDmMcCf]\b|--(?:delete|move|copy|force|set-upstream-to|unset-upstream))"
    r"|remote\s+(?:add|remove|rm|rename|set-url|set-head|prune))\b")
_GIT_READ = re.compile(
    _GIT + r"(?:status|diff|log|show|branch|remote|rev-parse|ls-files|ls-tree|ls-remote|blame|describe|grep"
    r"|shortlog|reflog|cat-file|merge-base|name-rev|for-each-ref|show-ref|check-ignore|count-objects"
    r"|config\s+(?:--get|--get-all|--list|-l))\b")
_FIND_DELETE = re.compile(r"^(?:\S*/)?find\b.*\s-(?:delete|fprint\w*|fls)\b", re.S)
_SORT_OUTPUT = re.compile(r"^sort\b.*\s(?:-o\s*\S|--output)", re.S)
_NETWORK_SEND = re.compile(
    r"^(?:\S*/)?(?:wget\b|(?:curl|http|https|xh)\b.*\s(?:-d|--data[\w-]*|-F|--form[\w-]*|-T|--upload-file|--json"
    r"|-X\s*(?!GET\b|HEAD\b)[A-Z]+|--request[= ](?!GET\b|HEAD\b)[A-Z]+|-o|-O|--output|--remote-name[\w-]*"
    r"|POST|PUT|PATCH|DELETE)\b)", re.S)
_CHECK_KINDS = frozenset({"read", "test", "network"})
_NO_EFFECT_WORDS = frozenset({
    # shell state and flow
    "cd", "pushd", "popd", "export", "unset", "set", "alias", "true", "false", ":", "test", "[", "[[", "sleep",
    "wait", "type", "hash", "command", "echo", "printf", "read",
    # stopping work is never gated
    "kill", "pkill", "killall",
    # reading files, processes and the system
    "pwd", "which", "whereis", "whoami", "id", "hostname", "uname", "date", "uptime", "env", "printenv", "locale",
    "cat", "ls", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "ag", "ack", "fd", "find", "tree",
    "wc", "stat", "file", "du", "df", "diff", "cmp", "comm", "sort", "uniq", "cut", "tr", "nl", "column", "fold",
    "fmt", "tac", "rev", "paste", "join", "seq", "basename", "dirname", "realpath", "readlink", "jq", "yq", "xxd",
    "od", "hexdump", "strings", "base64", "shasum", "sha1sum", "sha256sum", "sha512sum", "md5", "md5sum", "cksum",
    "ps", "pgrep", "lsof", "top", "free", "vm_stat", "sw_vers", "netstat", "dig", "nslookup", "host", "ping", "awk",
    "sed", "man",
})


def _changes(text: str) -> frozenset:
    """Deploys and file writes by the shared command patterns (redirects, cp, sed -i, installs, ...)."""
    kinds = matching_command_kinds(text)
    return frozenset(effect for kind, effect in (("deploy", DEPLOY), ("write", FILE_WRITE)) if kind in kinds)


def _nested(match: re.Match, depth: int) -> frozenset:
    body, rest = _unquote(match["body"])
    return _effects(body, depth + 1) | _changes(rest)


def _inline(match: re.Match, depth: int) -> frozenset:
    body, rest = _unquote(match["body"])
    return _code_effects(body, depth + 1) | _changes(rest)


def _fixed(*effects: str) -> Callable[[re.Match, int], frozenset]:
    found = frozenset(effects)
    return lambda _match, _depth: found


# Checked in order; the first that matches decides the simple command.
_STEP_RULES: tuple[tuple[re.Pattern, Callable[[re.Match, int], frozenset]], ...] = (
    (_SHELL_C, _nested),
    (_EVAL, _nested),
    (_FIND_EXEC, _nested),
    (_IN_PLACE, _fixed(FILE_WRITE)),
    (_INLINE, _inline),
    (_AGENT, _fixed(AGENT_LAUNCH)),
    (_MIGRATION, _fixed(MIGRATION)),
    (_GIT_PUSH, _fixed(REPO_CHANGE, DEPLOY)),  # a push can start a deploy
    (_GIT_TREE, _fixed(REPO_CHANGE, FILE_WRITE)),
    (_GIT_RECORD, _fixed(REPO_CHANGE)),
    (_FIND_DELETE, _fixed(FILE_WRITE)),
    (_SORT_OUTPUT, _fixed(FILE_WRITE)),
)


def _step_effects(step: str, depth: int) -> frozenset:
    step = _peel(step)
    if not step:
        return frozenset()
    for pattern, effects_of in _STEP_RULES:
        match = pattern.search(step)
        if match:
            return effects_of(match, depth)
    changes = _changes(step)
    if changes:
        return changes
    if _NETWORK_SEND.search(step):
        return frozenset({UNKNOWN})
    known = (matching_command_kinds(step) & _CHECK_KINDS or _command_word(step) in _NO_EFFECT_WORDS
             or _GIT_READ.search(step))
    return frozenset() if known else frozenset({UNKNOWN})


def _effects(command: str, depth: int) -> frozenset:
    if depth > _MAX_DEPTH:
        return frozenset({UNKNOWN})
    found: set[str] = set()
    for step in _split(command):
        found |= _step_effects(step, depth)
    return frozenset(found)


def command_effects(command: str) -> frozenset:
    """What a shell command line would change: members of ``STATE_CHANGES``, plus ``UNKNOWN`` for a
    step whose effect cannot be told; empty for reads, lookups, test runs and plain network checks."""
    return _effects(command or "", 0)


def command_changes_state(command: str) -> bool:
    """Whether any step would launch an agent, migrate data, change a repo, deploy or write a file."""
    return bool(command_effects(command) & STATE_CHANGES)


# -- inline scripts -------------------------------------------------------------------------------

_CODE_WRITE = re.compile(
    r"\bopen\s*\([^)]*?,\s*(?:mode\s*=\s*)?['\"][rbtU]*[wax+]"
    r"|\.write_(?:text|bytes)\s*\("
    r"|\b(?:os|shutil)\.(?:remove|unlink|rename|renames|replace|rmdir|removedirs|mkdir|makedirs|rmtree|move|copy\w*"
    r"|chmod|chown|symlink|link|truncate|utime)\s*\("
    r"|\.(?:unlink|rmdir|mkdir|touch|rename|replace|symlink_to|hardlink_to|chmod)\s*\("
    r"|\b(?:writeFile|appendFile|createWriteStream|copyFile|truncate|symlink|rm|cp)(?:Sync)?\s*\("
    r"|\bFile\.(?:write|delete|rename|unlink)\b|\bFileUtils\.")
_CODE_SPAWN = re.compile(
    r"\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*|posix_spawn\w*)\b|\bpty\.spawn\b|\bchild_process\b"
    r"|\b(?:execSync|execFileSync|spawnSync|execFile|spawn|system|popen|Popen)\s*\("
    r"|\bdo\s+shell\s+script\b|%x[{(\[]|\bDeno\.(?:run|Command)\b|\bBun\.spawn")
_LITERAL = re.compile(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"")
_ARGV = re.compile(r"[\[(]\s*((?:(?:'[^'\n]*'|\"[^\"\n]*\")\s*,\s*)+(?:'[^'\n]*'|\"[^\"\n]*\"))\s*,?\s*[\])]")


def _literals(text: str) -> list[str]:
    return [next(group for group in match.groups() if group is not None) for match in _LITERAL.finditer(text)]


def _code_effects(code: str, depth: int) -> frozenset:
    found = {FILE_WRITE} if _CODE_WRITE.search(code) else set()
    if _CODE_SPAWN.search(code):
        # A spawned command may be built at run time, so spawning is UNKNOWN whatever else is read.
        found.add(UNKNOWN)
        argvs = [" ".join(_literals(match[1])) for match in _ARGV.finditer(code)]
        for command in [*argvs, *_literals(code)]:
            found |= _effects(command, depth + 1)
    return frozenset(found)


def code_effects(code: str) -> frozenset:
    """What an inline script (``python -c``, ``node -e``, ``execute_code``) would change: file writes
    by their calls, and the commands it spawns read like any command line."""
    return _code_effects(code or "", 0)


# -- what an exit status of 0 shows ---------------------------------------------------------------

_LIST_OPS = frozenset({";", "\n", "&"})
_OPAQUE_OPS = frozenset({"(", ")", "$(", "`"})
_ERREXIT = re.compile(r"^set\s+(?:-\w*e\w*|-o\s+errexit)\b")
_PIPEFAIL = re.compile(r"^set\s+(?:-\w+\s+)*-o\s+pipefail\b|^set\s+-\w*o\w*\s+pipefail\b")


def _and_or_confirmed(pipelines: list[list[int]], ops: list[str], pipefail: bool) -> set[int]:
    """Steps of one ``&&``/``||`` list whose success its status 0 shows. The list is left-associative:
    ``… && last`` shows ``last`` and whatever the part before shows; ``… || last`` shows nothing, since
    ``last`` ran only if the part before failed. A pipeline shows its last command (all under pipefail)."""
    shown: set[int] = set()
    i = len(pipelines) - 1
    while True:
        if i > 0 and ops[i - 1] != "&&":
            return shown
        shown.update(pipelines[i] if pipefail else pipelines[i][-1:])
        if i == 0:
            return shown
        i -= 1


def _confirmed(parts: list[tuple[str, str]]) -> set[int]:
    """Positions in ``parts`` whose success an exit status of 0 for the whole line shows."""
    steps = [(position, step, op) for position, (step, op) in enumerate(parts) if step]
    if not steps or steps[-1][2] == "&":
        return set()  # nothing ran, or the last command runs in the background
    errexit = any(_ERREXIT.search(step) for _position, step, _op in steps)
    pipefail = any(_PIPEFAIL.search(step) for _position, step, _op in steps)
    lists: list[tuple[list[list[int]], list[str]]] = [([[]], [])]
    for position, step, op in steps:
        pipelines, ops = lists[-1]
        pipelines[-1].append(position)
        if op in ("&&", "||"):
            ops.append(op)
            pipelines.append([])
        elif op in _LIST_OPS:
            lists.append(([[]], []))
    lists = [(pipelines, ops) for pipelines, ops in lists if all(pipelines)]
    shown: set[int] = set()
    for pipelines, ops in (lists if errexit else lists[-1:]):  # under set -e a failing list ends the run
        shown |= _and_or_confirmed(pipelines, ops, pipefail)
    return shown


def _outcomes(command: str, shown_by_caller: bool, depth: int) -> list[tuple[str, bool]]:
    parts = _scan(command)
    opaque = any(op in _OPAQUE_OPS for _step, op in parts) or "<<" in command
    shown = set() if opaque or not shown_by_caller else _confirmed(parts)
    out: list[tuple[str, bool]] = []
    for position, (step, _op) in enumerate(parts):
        step = _peel(step)
        if not step:
            continue
        nested = _SHELL_C.search(step) or _EVAL.search(step)
        if nested and depth < _MAX_DEPTH:
            body, _rest = _unquote(nested["body"])
            out.extend(_outcomes(body, position in shown, depth + 1))
        else:
            out.append((step, position in shown))
    return out


def step_outcomes(command: str) -> tuple[tuple[str, bool], ...]:
    """The simple commands of a command line in order, each with whether an exit status of 0 for
    the whole line shows that it ran and succeeded. Only the last command of a pipeline (unless
    ``set -o pipefail``), and only along the trailing ``&&`` chain of the final list (every list
    under ``set -e``): ``a || b`` shows neither, ``a; b`` shows only ``b``. A nested ``bash -c`` is
    followed inside; subshells, substitutions and here-documents show nothing."""
    return tuple(_outcomes(command or "", True, 0))


def is_agent_run(step: str) -> bool:
    """Whether a simple command runs a coding agent, whose output is that agent's own report."""
    return bool(_AGENT_RUN.search(_peel(step)))

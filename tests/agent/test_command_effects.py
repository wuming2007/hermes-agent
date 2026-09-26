"""Behavior contract: what a shell command line or an inline script would change, however it is
wrapped. The evidence round and the working-memory scope gate both act on this answer, so a
rewrite that hides a command from one must hide it from neither.
"""

import pytest

from agent.command_effects import (
    AGENT_LAUNCH,
    DEPLOY,
    FILE_WRITE,
    MIGRATION,
    REPO_CHANGE,
    UNKNOWN,
    code_effects,
    command_changes_state,
    command_effects,
)


@pytest.mark.parametrize(
    "command, effects",
    [
        # a coding agent, however it is started
        pytest.param("claude -p 'fix it'", {AGENT_LAUNCH}, id="plain"),
        pytest.param('bash -c "claude -p hi"', {AGENT_LAUNCH}, id="bash-c"),
        pytest.param("sh -c 'cd repo && codex exec fix'", {AGENT_LAUNCH}, id="sh-c-with-a-chain"),
        pytest.param("eval 'claude -p hi'", {AGENT_LAUNCH}, id="eval"),
        pytest.param("npx -y @anthropic-ai/claude-code -p hi", {AGENT_LAUNCH}, id="npx-package"),
        pytest.param("echo `claude -p hi`", {AGENT_LAUNCH}, id="backticks"),
        pytest.param('echo "$(claude -p hi)"', {AGENT_LAUNCH}, id="substitution-inside-quotes"),
        pytest.param("echo hi | xargs -I {} claude -p {}", {AGENT_LAUNCH}, id="xargs"),
        pytest.param("echo start\nclaude -p hi", {AGENT_LAUNCH}, id="on-the-second-line"),
        pytest.param("sleep 1 & claude -p hi", {AGENT_LAUNCH}, id="after-a-background-job"),
        pytest.param("sudo env X=1 claude -p hi", {AGENT_LAUNCH}, id="sudo-then-env"),
        pytest.param("env X=1 sudo -u dev claude -p hi", {AGENT_LAUNCH}, id="env-then-sudo-as-a-user"),
        pytest.param("ANTHROPIC_MODEL=x claude -p hi", {AGENT_LAUNCH}, id="variable-assignment"),
        pytest.param("timeout -s KILL 600 claude -p hi", {AGENT_LAUNCH}, id="timeout"),
        pytest.param("(cd repo && nohup aider --yes) &", {AGENT_LAUNCH}, id="subshell-in-the-background"),
        # other changes
        pytest.param("bundle exec rake db:migrate", {MIGRATION}, id="migration-through-a-runner"),
        pytest.param("git -C repo commit -am wip", {REPO_CHANGE}, id="git-commit"),
        pytest.param("cd repo\ngit push origin main", {REPO_CHANGE, DEPLOY}, id="git-push"),
        pytest.param("git checkout -- parser.py", {REPO_CHANGE, FILE_WRITE}, id="git-checkout-rewrites-files"),
        pytest.param("./deploy.sh prod", {DEPLOY}, id="deploy-script"),
        pytest.param("find . -name '*.pyc' -delete", {FILE_WRITE}, id="find-delete"),
        pytest.param("find . -name '*.py' -exec sed -i s/a/b/ {} +", {FILE_WRITE}, id="find-exec"),
        pytest.param('bash -c "ls" > listing.txt', {FILE_WRITE}, id="nested-shell-output-to-a-file"),
        pytest.param("python3 -c \"open('out.txt','w').write('x')\"", {FILE_WRITE}, id="python-one-liner-write"),
        pytest.param("node -e \"require('fs').writeFileSync('a.txt','x')\"", {FILE_WRITE}, id="node-one-liner-write"),
        pytest.param("perl -pi -e 's/a/b/' parser.py", {FILE_WRITE}, id="perl-in-place"),
        # reads, checks and plain computation change nothing
        pytest.param("ls -la && git status", set(), id="reads"),
        pytest.param("grep -rn claude src/ | head -5", set(), id="searching-for-an-agent-name"),
        pytest.param("command -v claude; which codex", set(), id="looking-up-an-agent"),
        pytest.param("cd repo && pytest -q", set(), id="tests"),
        pytest.param("curl -s localhost:8080/health", set(), id="health-check"),
        pytest.param('git commit -m "docs; claude notes"', {REPO_CHANGE}, id="quoted-text-is-not-a-command"),
        pytest.param("python3 -c \"import json; print(json.dumps({'a': 1}))\"", set(), id="python-computation"),
        # effects that cannot be told from the command
        pytest.param("python3 refactor.py", {UNKNOWN}, id="script"),
        pytest.param("make build", {UNKNOWN}, id="build"),
        pytest.param("curl -X POST -d x=1 https://example.com/api", {UNKNOWN}, id="network-send"),
    ],
)
def test_what_a_command_would_change(command, effects):
    assert command_effects(command) == frozenset(effects)


def test_a_one_liner_that_spawns_a_command_is_read_like_that_command():
    effects = command_effects("python3 -c \"import subprocess; subprocess.run(['claude', '-p', 'fix it'])\"")

    assert AGENT_LAUNCH in effects


@pytest.mark.parametrize(
    "code, expected, unexpected",
    [
        pytest.param("from hermes_tools import read_file\nprint(read_file('a.py')['content'][:100])", set(),
                     {FILE_WRITE, UNKNOWN}, id="tool-calls-and-printing"),
        pytest.param("from pathlib import Path\nPath('a.py').write_text('x')", {FILE_WRITE}, set(), id="pathlib-write"),
        pytest.param("import shutil\nshutil.rmtree('build')", {FILE_WRITE}, set(), id="remove-a-tree"),
        pytest.param("with open('notes.md', 'a') as f:\n    f.write('x')", {FILE_WRITE}, set(), id="append"),
        pytest.param("with open('notes.md') as f:\n    print(f.read())", set(), {FILE_WRITE}, id="read"),
        pytest.param("import os\nos.system('git push origin main')", {REPO_CHANGE, DEPLOY}, set(), id="os-system"),
        pytest.param("import subprocess\nsubprocess.run(['git', 'push', 'origin', 'main'])", {REPO_CHANGE, DEPLOY},
                     set(), id="subprocess-argv"),
    ],
)
def test_what_an_inline_script_would_change(code, expected, unexpected):
    effects = code_effects(code)

    assert expected <= effects
    assert not unexpected & effects


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("bash " + "--x " * 26 + "script.sh", id="many-shell-options"),
        pytest.param("python3 " + "--x " * 26 + "script.py", id="many-interpreter-options"),
        pytest.param("timeout " + "-s " * 26 + "x", id="many-timeout-options"),
    ],
)
def test_reading_a_command_takes_linear_time(command):
    import time

    started = time.perf_counter()
    command_effects(command)

    assert time.perf_counter() - started < 2.0


@pytest.mark.parametrize(
    "command, changes",
    [
        pytest.param("pytest -q && cat report.txt", False, id="check"),
        pytest.param("python3 app.py --help", False, id="running-the-program"),
        pytest.param("bash -c 'claude -p \"fix the tests\"'", True, id="hidden-agent"),
        pytest.param("timeout 60 alembic upgrade head", True, id="migration"),
        pytest.param("echo ok\ntouch done.flag", True, id="write-on-a-later-line"),
    ],
)
def test_a_state_change_is_any_launch_migration_repo_change_deploy_or_write(command, changes):
    assert command_changes_state(command) is changes


@pytest.mark.parametrize(
    "command, outcomes",
    [
        pytest.param("pytest -q", [("pytest -q", True)], id="single"),
        pytest.param("pytest -q && ./deploy.sh", [("pytest -q", True), ("./deploy.sh", True)], id="and-chain"),
        pytest.param("true || pytest tests/test_beta.py", [("true", False), ("pytest tests/test_beta.py", False)],
                     id="or-may-have-skipped"),
        pytest.param("pytest -q || true", [("pytest -q", False), ("true", False)], id="or-masks-a-failure"),
        pytest.param("(false || make) && pytest -q", [("false", False), ("make", False), ("pytest -q", False)],
                     id="subshell-shows-nothing"),
        pytest.param("pytest -q 2>&1 | tail -5", [("pytest -q 2>&1", False), ("tail -5", True)],
                     id="pipe-shows-its-last-command"),
        pytest.param("set -o pipefail; pytest -q | tee log.txt",
                     [("set -o pipefail", False), ("pytest -q", True), ("tee log.txt", True)], id="pipefail"),
        pytest.param("pytest -q; ./deploy.sh", [("pytest -q", False), ("./deploy.sh", True)],
                     id="semicolon-shows-the-last"),
        pytest.param("set -e; pytest -q; ./deploy.sh", [("set -e", True), ("pytest -q", True), ("./deploy.sh", True)],
                     id="errexit"),
        pytest.param("make build || make fallback && pytest -q",
                     [("make build", False), ("make fallback", False), ("pytest -q", True)], id="left-associative"),
        pytest.param("bash -c 'cd app && pytest -q'", [("cd app", True), ("pytest -q", True)], id="nested-shell"),
        pytest.param("timeout 300 pytest -q", [("pytest -q", True)], id="wrapper-peeled"),
        pytest.param("pytest -q &", [("pytest -q", False)], id="background"),
        pytest.param("cat <<EOF > notes.txt\nhello\nEOF", [("cat <<EOF > notes.txt", False), ("hello", False),
                                                          ("EOF", False)], id="here-document-shows-nothing"),
    ],
)
def test_what_an_exit_status_of_zero_shows(command, outcomes):
    from agent.command_effects import step_outcomes

    assert list(step_outcomes(command)) == outcomes

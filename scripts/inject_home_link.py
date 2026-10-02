from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

MARK_BEGIN = "<!-- rlforge:home-link -->"
MARK_END = "<!-- /rlforge:home-link -->"

BUTTON_TEMPLATE = MARK_BEGIN + """
<a id="rlforge-samples-link" href="__HOME_URL__#samples" title="查看训练 rollout 与评测抽样（reward / 完整回答）"
   style="position:fixed;right:18px;bottom:64px;z-index:99999;display:inline-flex;align-items:center;gap:7px;padding:9px 14px;border-radius:999px;background:#0e9f6e;color:#fff;font:650 13px/1 -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;text-decoration:none;box-shadow:0 6px 18px rgba(24,35,56,.22);">&#9776; Rollout 样本</a>
<a id="rlforge-home-link" href="__HOME_URL__" title="跳回实验主页"
   style="position:fixed;right:18px;bottom:18px;z-index:99999;display:inline-flex;align-items:center;gap:7px;padding:9px 14px;border-radius:999px;background:#3858d6;color:#fff;font:650 13px/1 -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;text-decoration:none;box-shadow:0 6px 18px rgba(24,35,56,.22);">&#8962; 实验主页</a>
<script>
(function () {
  var HUB = "__HOME_URL__";
  function retarget() {
    var a = document.getElementById("rlforge-samples-link");
    if (!a) return;
    fetch(HUB + "api/samples_runs").then(function (r) { return r.json(); }).then(function (runs) {
      var ids = runs.map(function (x) { return x.id || x; });
      var txt = document.body ? document.body.innerText : "";
      var hits = ids.filter(function (id) { return txt.indexOf(id) >= 0; });
      if (hits.length === 1) {
        a.href = HUB + "samples/" + hits[0];
        a.innerHTML = "&#9776; 本 run Rollout 样本";
        a.title = hits[0] + " 的训练/评测抽样";
      } else {
        a.href = HUB + "#samples";
        a.innerHTML = "&#9776; Rollout 样本";
        a.title = "查看训练 rollout 与评测抽样（reward / 完整回答）";
      }
    }).catch(function () {});
  }
  retarget();
  setInterval(retarget, 3000);
})();
</script>
""" + MARK_END


def find_template() -> Path:
    spec = importlib.util.find_spec("swanboard")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("swanboard package not found; run with the panel venv's python")
    template = Path(spec.submodule_search_locations[0]) / "template" / "index.html"
    if not template.is_file():
        raise SystemExit(f"swanboard template not found: {template}")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject a back-to-home button into the SwanLab panel template (idempotent)."
    )
    parser.add_argument("--home-url", default="http://127.0.0.1:63400/")
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--check", action="store_true", help="only verify the link is present")
    args = parser.parse_args()

    template = args.template or find_template()
    html = template.read_text(encoding="utf-8")
    block = BUTTON_TEMPLATE.replace("__HOME_URL__", args.home_url)

    if MARK_BEGIN in html:
        if args.check:
            return
        start = html.index(MARK_BEGIN)
        end = html.index(MARK_END) + len(MARK_END)
        html = html[:start] + block + html[end:]
    else:
        if args.check:
            raise SystemExit(f"home link not injected: {template}")
        if "</body>" in html:
            html = html.replace("</body>", block + "\n</body>", 1)
        else:
            html = html + "\n" + block + "\n"

    template.write_text(html, encoding="utf-8")
    print(f"[rlforge home-link] injected into {template} -> {args.home_url}")


if __name__ == "__main__":
    main()

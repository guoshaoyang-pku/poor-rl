#!/usr/bin/env python3
"""Inject a drag-resizable left experiment sidebar into the swanboard UI.

SwanLab's bundled sidebar has a fixed width, truncating long experiment
names. This patch appends a small script to swanboard's index.html that
locates the left experiment column at runtime, attaches a drag handle and
persists the chosen width in localStorage. Idempotent (marker-guarded);
keeps a index.html.orig backup next to the patched file.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

MARKER = "/*rlforge-sidebar-resize*/"

SNIPPET = """<script>/*rlforge-sidebar-resize*/
(function(){
  var curW=+(localStorage.getItem('swanSidebarW')||0);
  var sb=null,handle=null,obs=null;
  function cand(){
    var els=[].slice.call(document.querySelectorAll('div,aside,nav,section'));
    var hit=els.filter(function(el){
      var r=el.getBoundingClientRect();
      if(r.width<140||r.width>560||r.left>90||r.height<window.innerHeight*0.5)return false;
      var s=getComputedStyle(el);
      return s.overflowY==='auto'||s.overflowY==='scroll'||el.scrollHeight>el.clientHeight+4;
    });
    hit.sort(function(a,b){return a.getBoundingClientRect().width-b.getBoundingClientRect().width;});
    return hit[0]||null;
  }
  function apply(w){ if(!sb)return; curW=w;
    sb.style.width=w+'px'; sb.style.minWidth=w+'px'; sb.style.maxWidth=w+'px';
    sb.style.flex='0 0 '+w+'px'; }
  function place(){ if(!sb||!handle)return;
    var r=sb.getBoundingClientRect(); handle.style.left=(r.right-3)+'px'; }
  function install(){
    if(sb)return; sb=cand(); if(!sb)return;
    handle=document.createElement('div');
    handle.style.cssText='position:fixed;top:0;bottom:0;width:7px;cursor:col-resize;z-index:99999;';
    document.body.appendChild(handle);
    place();
    obs=new MutationObserver(place);
    obs.observe(document.body,{childList:true,subtree:true});
    window.addEventListener('resize',place);
    var drag=null;
    handle.addEventListener('mousedown',function(e){drag={x:e.clientX,w:sb.getBoundingClientRect().width};e.preventDefault();});
    window.addEventListener('mousemove',function(e){ if(!drag)return;
      var w=Math.min(Math.max(180,Math.round(drag.w+(e.clientX-drag.x))),Math.round(window.innerWidth*0.75));
      apply(w); localStorage.setItem('swanSidebarW',String(w)); place(); });
    window.addEventListener('mouseup',function(){drag=null;});
    if(curW)apply(curW);
    setInterval(function(){ if(curW)apply(curW); },1000);
  }
  var t=setInterval(function(){install(); if(sb)clearInterval(t);},400);
  setTimeout(function(){clearInterval(t);},60000);
})();
</script>
"""


def main() -> int:
    if len(sys.argv) > 1:
        index = Path(sys.argv[1])
    else:
        import swanboard  # type: ignore

        index = Path(swanboard.__file__).resolve().parent / "template" / "index.html"
    if not index.is_file():
        print(f"[patch] index.html not found: {index}", file=sys.stderr)
        return 1
    text = index.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[patch] already patched: {index}")
        return 0
    backup = index.with_name(index.name + ".orig")
    if not backup.exists():
        shutil.copy2(index, backup)
    text = text.replace("</body>", SNIPPET + "</body>", 1)
    if MARKER not in text:  # no </body> tag: append at EOF
        text += SNIPPET
    index.write_text(text, encoding="utf-8")
    print(f"[patch] sidebar resize injected: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

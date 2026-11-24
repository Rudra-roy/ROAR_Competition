/* Site behaviour: nav state, hero scroll choreography, video, scroll reveals, lazy data loading.
   Classic script, no modules or fetch, so it works from file:// as well as on GitHub Pages. */
(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.add("js");
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var ROAR = (window.ROAR = window.ROAR || {});
  var loaded = {};

  /* load a data script once; cb runs when it is available (also if it was already loaded) */
  ROAR.load = function (src, cb) {
    var e = loaded[src];
    if (e && e.done) return cb && cb();
    if (e) { if (cb) e.cbs.push(cb); return; }
    e = loaded[src] = { done: false, cbs: cb ? [cb] : [] };
    var s = document.createElement("script");
    s.src = src;
    s.onload = function () { e.done = true; e.cbs.forEach(function (f) { f(); }); };
    s.onerror = function () { console.warn("could not load " + src); };
    document.head.appendChild(s);
  };

  /* run fn once, when el is within `margin` of the viewport */
  ROAR.whenNear = function (el, fn, margin) {
    if (!el) return;
    if (!("IntersectionObserver" in window)) return fn();
    var io = new IntersectionObserver(function (es) {
      if (es.some(function (e) { return e.isIntersecting; })) { io.disconnect(); fn(); }
    }, { rootMargin: (margin || 600) + "px 0px" });
    io.observe(el);
  };

  var hero = document.querySelector(".hero");
  var body = document.body;

  /* nav: transparent over the hero, solid after it */
  function navState() {
    var past = hero ? hero.getBoundingClientRect().bottom < 80 : true;
    body.classList.toggle("scrolled", past);
  }

  /* hero: --p drives scale/scrim/headline, from 0 to 1 over the first 80% of the viewport height */
  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      navState();
      if (hero && !reduce) {
        var p = Math.min(1, Math.max(0, window.scrollY / (0.8 * window.innerHeight)));
        var q = Math.min(1, Math.max(0, (p - 0.55) / 0.35));
        hero.style.setProperty("--p", p.toFixed(4));
        hero.style.setProperty("--p2", (q * q * (3 - 2 * q)).toFixed(4));
      }
      evo();
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  if (!hero) body.classList.add("plain");

  /* hero video: plays muted unless reduced motion or data saving; pauses off-screen and when the tab is hidden */
  var video = document.querySelector(".hero video");
  var vbtn = document.querySelector(".vbtn");
  var userPaused = false;
  function setBtn() {
    if (!vbtn || !video) return;
    vbtn.setAttribute("aria-label", video.paused ? "Play video" : "Pause video");
    vbtn.innerHTML = video.paused
      ? '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5v11l9-5.5z" fill="currentColor"/></svg>'
      : '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5h3v11H3zM8 1.5h3v11H8z" fill="currentColor"/></svg>';
  }
  if (video) {
    var saver = navigator.connection && navigator.connection.saveData;
    if (!reduce && !saver) {
      video.preload = "auto";
      var tryPlay = function () { var p = video.play(); if (p && p.catch) p.catch(function () {}); };
      tryPlay();
      video.addEventListener("play", setBtn);
      video.addEventListener("pause", setBtn);
    }
    if (vbtn) {
      setBtn();
      vbtn.addEventListener("click", function () {
        if (video.paused) { userPaused = false; video.play(); } else { userPaused = true; video.pause(); }
      });
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (reduce || userPaused) return;
          if (e.isIntersecting) { var p = video.play(); if (p && p.catch) p.catch(function () {}); } else video.pause();
        });
      }).observe(hero);
    }
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) video.pause(); else if (!reduce && !userPaused && !saver) video.play().catch(function () {});
    });
  }

  /* reveal on scroll: once, transform and opacity only, staggered by --i */
  var rv = [].slice.call(document.querySelectorAll(".rv"));
  if (reduce || !("IntersectionObserver" in window)) rv.forEach(function (e) { e.classList.add("in"); });
  else {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    rv.forEach(function (e) { io.observe(e); });
  }

  /* evolution timeline: the red line advances with the scroll, steps light up as it passes them */
  var steps = document.querySelector(".steps");
  function evo() {
    if (!steps) return;
    var r = steps.getBoundingClientRect(), vh = window.innerHeight;
    var prog = reduce ? 1 : Math.min(1, Math.max(0, (vh * 0.78 - r.top) / Math.max(r.height, 1)));
    steps.style.setProperty("--prog", prog.toFixed(3));
    var items = steps.children, wide = window.innerWidth >= 1080;
    for (var i = 0; i < items.length; i++) {
      var f = (i + 0.5) / items.length;
      items[i].classList.toggle("on", prog >= (wide ? i / items.length + 0.02 : f - 0.08));
    }
  }

  /* mobile menu: Esc closes and returns focus, a link click closes */
  var menu = document.querySelector("details.menu");
  if (menu) {
    menu.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && menu.open) { menu.open = false; menu.querySelector("summary").focus(); }
    });
    menu.addEventListener("click", function (e) { if (e.target.closest("a")) menu.open = false; });
    menu.addEventListener("toggle", function () { menu.querySelector("summary").setAttribute("aria-expanded", menu.open); });
  }

  /* book: scroll-spy for the sidebar */
  var side = document.querySelector(".sidebar");
  if (side && "IntersectionObserver" in window) {
    var links = {};
    [].forEach.call(side.querySelectorAll("a"), function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var heads = [].slice.call(document.querySelectorAll(".chapter > h2, .chapter h3[id]"));
    var cur = null;
    var spy = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        var id = e.target.tagName === "H2" ? e.target.parentNode.id : e.target.id; // chapter headings: the section holds the id
        if (e.isIntersecting && links[id]) {
          if (cur) cur.removeAttribute("aria-current");
          cur = links[id]; cur.setAttribute("aria-current", "location");
        }
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    heads.forEach(function (h) { spy.observe(h); });
  }

  onScroll();
  navState();
})();

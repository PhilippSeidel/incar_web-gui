// Links a demo's videos and charts in the dataset inspector: a time cursor in every chart follows the video,
// and clicking inside a chart moves the cursor there and seeks every video to that moment.
// videos: [{id, times, overlay}] with times = capture time (s since demo start) of each video frame and overlay the
// element that shows the text features (e.g. the prompt) active at that moment; charts: element ids;
// texts: {feature: {t, text, colors}} (colors: text -> its timeline colour), replaced by demoSetTexts() on edits.
window.linkDemo = (videos, charts, fps, texts = {}) => {
  let now = 0;
  const textAt = ({t, text}, at) => {  // the value recorded last at or before `at`
    let lo = 0, hi = t.length - 1, found = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (t[mid] <= at) { found = mid; lo = mid + 1; } else hi = mid - 1; }
    return found < 0 ? (text[0] ?? '') : text[found];  // before the first sample: the first recorded text
  };
  const caption = (at) => {  // one line per text feature, in the colour of that text on the timeline
    const names = Object.keys(texts);
    return names.map((name) => {
      const text = textAt(texts[name], at), line = document.createElement('div');
      line.textContent = (names.length > 1 ? name + ': ' : '') + text;
      line.style.background = (texts[name].colors?.[text] ?? '#64748b') + '8c';  // 55 % like the timeline segments
      return line;
    });
  };
  const ready = (id, get, then, tries = 100) => {  // NiceGUI mounts elements (and ECharts inits) asynchronously
    const value = get(getElement(id));
    if (value) then(value);
    else if (tries > 0) setTimeout(() => ready(id, get, then, tries - 1), 50);
  };
  const cursor = (t) => ({series: [{markLine: {
    silent: true, symbol: 'none', animation: false, label: {show: false}, z: 10,  // above a text timeline's segments
    lineStyle: {color: '#f8fafc', width: 1.5, type: 'solid'}, data: [{xAxis: t}],
  }}]});
  const draw = (t) => {
    now = t;
    charts.forEach((id) => getElement(id)?.chart?.setOption(cursor(t)));
    videos.forEach(({overlay}) => { const el = overlay && getHtmlElement(overlay); if (el) el.replaceChildren(...caption(t)); });
  };
  // Two video layouts exist: frame i at i / fps (when the camera kept the dataset fps), or every frame at its capture
  // time (newer recordings; the camera may run slower or unevenly). The video's length tells them apart.
  const realTime = (video, times) => times.length > 1 && video.duration > 0 &&
    Math.abs(video.duration - (times[times.length - 1] - times[0])) < Math.max(0.5, 0.03 * video.duration);
  // frame on screen -> its capture time, and back (first frame captured at or after t)
  const demoTime = (video, times) => {
    if (!times.length) return video.currentTime;
    if (realTime(video, times)) return video.currentTime + times[0];
    return times[Math.min(times.length - 1, Math.max(0, Math.floor(video.currentTime * fps)))];
  };
  const frameStart = (video, times, t) => {
    if (!times.length) return t;
    if (realTime(video, times)) return Math.max(0, t - times[0]);
    let lo = 0, hi = times.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] < t) lo = mid + 1; else hi = mid; }
    return (lo + 0.5) / fps;  // middle of the frame, so rounding never lands on the previous one
  };

  const seek = (t) => {
    draw(t);
    for (const {id: videoId, times} of videos) {
      const video = getElement(videoId)?.$el;
      if (video) video.currentTime = frameStart(video, times, t);
    }
  };
  window.demoSeek = seek;
  window.demoSetTexts = (name, text, colors) => { if (texts[name]) { Object.assign(texts[name], {text, colors}); draw(now); } };
  draw(0);

  for (const {id, times} of videos) {
    ready(id, (el) => el?.$el, (video) => {
      let frame = 0;
      const tick = () => {
        draw(demoTime(video, times));
        if (!video.paused) frame = requestAnimationFrame(tick);
      };
      video.addEventListener('play', () => { cancelAnimationFrame(frame); tick(); });
      video.addEventListener('seeked', tick);
      video.addEventListener('loadeddata', tick);
    });
  }
  for (const id of charts) {
    ready(id, (el) => el?.chart, (chart) => chart.getZr().on('click', (e) => {
      const point = [e.offsetX, e.offsetY];
      if (!chart.containPixel({gridIndex: 0}, point)) return;  // ignore legend and zoom slider clicks
      const t = chart.convertFromPixel({gridIndex: 0}, point)[0];  // grid finder returns [x, y]
      seek(t);
      emitEvent('demo_chart_click', {chart: id, t});  // e.g. a text timeline selects the segment at t
    }));
  }
};

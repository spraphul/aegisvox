/* Progressive enhancement: native video controls work without JavaScript. */
(() => {
  const video = document.querySelector('#project-film');
  if (!video) return;
  const play = document.querySelector('#film-play');
  const status = document.querySelector('#film-status');
  const pauseOthers = () => document.querySelectorAll('audio, video').forEach(media => {
    if (media !== video) media.pause();
  });
  async function playWithSound(time) {
    try {
      if (typeof time === 'number') video.currentTime = time;
      pauseOthers();
      video.muted = false;
      video.volume = 1;
      await video.play();
      status.textContent = 'Sound on · No subtitles';
    } catch (_) {
      status.textContent = 'Use the video controls, or download the MP4 below.';
    }
  }
  play.addEventListener('click', () => video.paused ? playWithSound() : video.pause());
  document.querySelectorAll('[data-film-time]').forEach(button => {
    button.addEventListener('click', () => playWithSound(Number(button.dataset.filmTime)));
  });
  video.addEventListener('play', () => { pauseOthers(); play.textContent = 'Pause film'; });
  video.addEventListener('pause', () => { play.textContent = 'Play with sound'; });
  video.addEventListener('ended', () => { status.textContent = 'Film complete · Replay any chapter'; });
  video.addEventListener('error', () => { status.textContent = 'Video could not load. Try the download link.'; });
})();

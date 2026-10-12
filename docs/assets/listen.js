/* Native controls remain usable without JavaScript. No analytics or libraries. */
(() => {
  const gallery = document.querySelector('#listen');
  if (!gallery) return;
  const cards = [...gallery.querySelectorAll('.audio-example')];
  const filters = [...gallery.querySelectorAll('select')];
  const count = gallery.querySelector('#example-count');

  function applyFilters() {
    const dataset = gallery.querySelector('#dataset-filter').value;
    const language = gallery.querySelector('#language-filter').value;
    let shown = 0;
    cards.forEach(card => {
      const visible = (dataset === 'all' || dataset === card.dataset.corpus) &&
        (language === 'all' || language === card.dataset.language);
      card.hidden = !visible;
      if (visible) shown++;
      else card.querySelectorAll('audio').forEach(audio => audio.pause());
    });
    count.textContent = `${shown} of ${cards.length} conversations`;
    gallery.querySelector('#no-examples').hidden = shown !== 0;
  }
  filters.forEach(filter => filter.addEventListener('change', applyFilters));
  gallery.querySelector('#reset-filters').addEventListener('click', () => {
    filters.forEach(filter => { filter.value = 'all'; });
    applyFilters();
  });
  document.addEventListener('play', event => {
    if (!(event.target instanceof HTMLMediaElement)) return;
    document.querySelectorAll('audio, video').forEach(media => {
      if (media !== event.target) media.pause();
    });
  }, true);
  gallery.querySelectorAll('audio').forEach(audio => {
    const status = audio.parentElement.querySelector('.audio-status');
    audio.addEventListener('error', () => {
      status.textContent = 'Audio could not load. Use Open MP3 below, or retry.';
      status.hidden = false;
      audio.parentElement.querySelector('.retry-audio').hidden = false;
    });
    audio.addEventListener('loadedmetadata', () => {
      status.hidden = true;
      audio.parentElement.querySelector('.retry-audio').hidden = true;
    });
    audio.parentElement.querySelector('.retry-audio').addEventListener('click', () => audio.load());
  });
})();

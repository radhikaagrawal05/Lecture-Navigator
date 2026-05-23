chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'jump') {
    const video = document.querySelector('video');
    if (video) {
      video.currentTime = request.time;
      video.play();
    }
  }
});

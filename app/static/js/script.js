function clicked_img(img) {
    var top = document.getElementById('top');
    var overlay = document.getElementById('overlay');

    top.src = img.src;
    overlay.style.display = 'block';

    if (img.naturalWidth < window.innerWidth * 0.6 && img.naturalHeight < window.innerHeight * 0.6) {
        top.width = img.naturalWidth;
        top.height = img.naturalHeight;
    } else {
        top.width = window.innerWidth * 0.6;
        top.height = img.naturalHeight / img.naturalWidth * top.width;
    }
}

function do_close() {
    document.getElementById('overlay').style.display = 'none';
}

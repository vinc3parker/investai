let lastScrollTop = 0;
const header = document.querySelector("header");

window.addEventListener("scroll", function () {
  let scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  if (scrollTop > lastScrollTop) {
    // User is scrolling down, hide the header
    header.style.top = "-80px"; // Adjust based on your header's height
  } else {
    // User is scrolling up, show the header
    header.style.top = "0";
  }
  lastScrollTop = scrollTop;
});

document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("stockChart").getContext("2d");
  const ticker = "AAPL"; // Example ticker, you might want to make this dynamic

  fetch(`http://localhost:5000/fetch/${ticker}`) // Adjust the URL as needed
    .then((response) => response.json())
    .then((data) => {
      const dates = data.map((item) => item.Date);
      const prices = data.map((item) => item.Close); // Example using 'Close' price

      const chart = new Chart(ctx, {
        type: "line", // 'candlestick' for financial charts if using a plugin
        data: {
          labels: dates,
          datasets: [
            {
              label: `${ticker} Stock Price`,
              data: prices,
              borderColor: "rgb(75, 192, 192)",
              tension: 0.1,
            },
          ],
        },
        options: {
          scales: {
            x: {
              type: "time",
              time: {
                unit: "day",
              },
            },
            y: {
              beginAtZero: false,
            },
          },
        },
      });
    })
    .catch((error) => console.error("Error loading the stock data:", error));
});

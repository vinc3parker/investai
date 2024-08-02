document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("stockChart").getContext("2d");
  const ticker = "AAPL"; // Adjust as needed

  fetch(`http://127.0.0.1:5000/fetch/${ticker}`)
    .then((response) => response.json())
    .then((data) => {
      if (!data || data.length === 0) {
        console.error("No data or empty data received");
        ctx.fillText(
          "No data available",
          ctx.canvas.width / 2,
          ctx.canvas.height / 2
        );
        return;
      }

      const financialData = data.map((item) => ({
        t: new Date(item.date), // Date
        o: item.open, // Open
        h: item.high, // High
        l: item.low, // Low
        c: item.close, // Close
      }));

      const chart = new Chart(ctx, {
        type: "candlestick",
        data: {
          datasets: [
            {
              label: `${ticker} Stock Price`,
              data: financialData,
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
              title: {
                display: true,
                text: "Price",
              },
            },
          },
        },
      });

      console.log("Candlestick chart has been created");
    })
    .catch((error) => {
      console.error("Error loading the stock data:", error);
      ctx.fillText(
        "Failed to fetch data",
        ctx.canvas.width / 2,
        ctx.canvas.height / 2
      );
    });
});

document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("stockChart").getContext("2d");

  if (!ctx) {
    console.error("Failed to get context");
    return;
  }

  console.log("Canvas context acquired");

  const ticker = "AAPL"; // Example ticker, adjust as needed
  console.log("Fetching data for ticker:", ticker);

  fetch(`http://127.0.0.1:5000/fetch/${ticker}`) // Ensure this URL is correct
    .then((response) => {
      console.log("API response received");
      return response.json();
    })
    .then((data) => {
      console.log("Data received from API:", JSON.stringify(data));

      if (!data || data.length === 0) {
        console.error("No data or empty data received");
        ctx.fillText(
          "No data available",
          ctx.canvas.width / 2,
          ctx.canvas.height / 2
        );
        return;
      }

      const labels = data.map((item) =>
        new Date(item.date).toLocaleDateString("en-US")
      );
      const closePrices = data.map((item) => item.close);

      const chart = new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: `${ticker} Closing Prices`,
              data: closePrices,
              borderColor: "rgb(75, 192, 192)",
              backgroundColor: "rgba(75, 192, 192, 0.5)",
              fill: false,
            },
          ],
        },
        options: {
          scales: {
            x: {
              type: "time",
              time: {
                unit: "day",
                tooltipFormat: "ll",
              },
              title: {
                display: true,
                text: "Date",
              },
            },
            y: {
              beginAtZero: false,
              title: {
                display: true,
                text: "Closing Price ($)",
              },
            },
          },
        },
      });

      console.log("Chart has been created");
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

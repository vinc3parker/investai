// Define variables for chart context and chart instance
var ctx = document.getElementById("chart").getContext("2d");
ctx.canvas.width = 1000;
ctx.canvas.height = 250;

var chart; // Variable to hold the chart instance

// Function to fetch and populate tickers into the select dropdown
async function loadTickers() {
  const tickerAPIUrl = "http://127.0.0.1:5000/tickers";
  const response = await fetch(tickerAPIUrl);
  const tickers = await response.json();
  const tickerSelector = document.getElementById("ticker-select");

  tickers.forEach((ticker) => {
    const option = document.createElement("option");
    option.value = ticker;
    option.textContent = ticker;
    tickerSelector.appendChild(option);
  });
}

// Function to create and update the chart with real data
async function createChart(selectedTicker) {
  const response = await fetch(`/api/ticker-data/${selectedTicker}/`);
  const rawData = await response.json();

  const data = rawData.map((item) => {
    return {
      x: new Date(item.date).getTime(), // Convert the date string to a JavaScript Date object
      o: item.open,
      h: item.high,
      l: item.low,
      c: item.close,
    };
  });

  if (!chart) {
    // If chart doesn't exist, create it
    chart = new Chart(ctx, {
      type: "candlestick",
      data: {
        datasets: [
          {
            label: selectedTicker,
            data: data, // Use real data fetched from the API
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
  } else {
    // If chart already exists, just update it with the new data
    chart.data.datasets[0].label = selectedTicker;
    chart.data.datasets[0].data = data;
    chart.update();
  }
}

// Function to update the chart with the selected time range
function updateChartWithRange(range) {
    const selectedTicker = document.getElementById('ticker-select').value;
    if (selectedTicker) {
        createChart(selectedTicker, range);
    }
}

// Event listener for loading the tickers when the page loads
document.addEventListener("DOMContentLoaded", function () {
  loadTickers(); // Populate the ticker selector

  // Event listener for when a ticker is selected
  document
    .getElementById("ticker-select")
    .addEventListener("change", function () {
      const selectedTicker = this.value;
      if (selectedTicker) {
        createChart(selectedTicker); // Create or update the chart with the selected ticker's data
      }
    });
});

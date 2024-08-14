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

  // Fetch current stock value (assuming the last item in the data array is the current value)
  const currentValue = data[data.length - 1].c;
  document.getElementById("current-value").textContent = currentValue;

  // Fetch predicted value from the backend
  const predictionResponse = await fetch(`/api/predict/${selectedTicker}/`);
  const predictionData = await predictionResponse.json();
  const predictedValue = predictionData.predicted_value;

  if (predictedValue !== null) {
    document.getElementById("predicted-value").textContent = predictedValue;

    // Calculate the expected change
    const expectedChange = predictedValue - currentValue;
    document.getElementById("expected-change").textContent = expectedChange;

    // Calculate expected profit percentage for long and short positions
    const profitLong = ((predictedValue - currentValue) / currentValue) * 100;
    const profitShort = ((currentValue - predictedValue) / currentValue) * 100;

    document.getElementById("profit-long").textContent =
      profitLong.toFixed(2) + "%";
    document.getElementById("profit-short").textContent =
      profitShort.toFixed(2) + "%";
  } else {
    document.getElementById("predicted-value").textContent =
      "Prediction not available";
    document.getElementById("expected-change").textContent = "-";
    document.getElementById("profit-long").textContent = "-";
    document.getElementById("profit-short").textContent = "-";
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

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ABSA Sentiment Web</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <h1>🧠 ABSA Sentiment Dashboard</h1>
  <textarea id="review" placeholder="Enter a review..." rows="4" cols="60"></textarea><br>
  <button id="analyzeBtn">Analyze</button>
  <div id="result"></div>

  <script>
    document.getElementById("analyzeBtn").addEventListener("click", async () => {
      const text = document.getElementById("review").value;
      if (!text.trim()) {
        alert("Please enter a review.");
        return;
      }

      document.getElementById("result").innerHTML = "⏳ Processing...";
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      if (data.error) {
        document.getElementById("result").innerHTML = `❌ Error: ${data.error}`;
        return;
      }

      let html = "<h3>Results:</h3><table border='1'><tr><th>Clause</th><th>Term</th><th>Opinion</th><th>Category</th><th>Polarity</th></tr>";
      data.forEach(row => {
        html += `<tr>
          <td>${row.clause || ""}</td>
          <td>${row.term || ""}</td>
          <td>${row.opinion || ""}</td>
          <td>${row.category || ""}</td>
          <td>${row.polarity || ""}</td>
        </tr>`;
      });
      html += "</table>";
      document.getElementById("result").innerHTML = html;
    });
  </script>
</body>
</html>

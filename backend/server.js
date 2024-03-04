const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");

const app = express();
const port = 3000;

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET, POST, OPTIONS, PUT, DELETE"
  );
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  next();
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(null, "upload.jpg");
  },
});

const upload = multer({ storage: storage });

app.post("/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  // Run the Python script using child_process
  exec("python temp.py", (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error}`);
      console.error(`Python script stderr: ${stderr}`);
      return res.status(500).json({ error: "Internal Server Error" });
    }

    console.log(`Python script output: ${stdout}`);
    res.json({ message: "File uploaded successfully" });
  });
});

app.use(express.static(path.join(__dirname, "processed")));

// Define endpoint to serve the image file
app.get("/image", async (req, res) => {
  const imagePath = path.join(__dirname, "processed", "restored.jpg");
  const checkImage = () => {
    return new Promise((resolve, reject) => {
      fs.access(imagePath, fs.constants.F_OK, (err) => {
        if (!err) {
          resolve(true); // Image exists
        } else {
          reject(false); // Image does not exist
        }
      });
    });
  };

  try {
    // Check if the image exists
    await checkImage();
    // If the image exists, send it
    res.sendFile(imagePath);
  } catch (error) {
    // If the image does not exist, wait and try again after a delay
    setTimeout(() => {
      res.redirect("/image");
    }, 5000); // Wait for 5 seconds before trying again
  }
});

app.delete("/delete-image", async (req, res) => {
  try {
    const imagePath = path.join(__dirname, "processed", "restored.jpg");

    // Check if the file exists before attempting to delete
    const fileExists = await fs.promises
      .access(imagePath, fs.constants.F_OK)
      .then(() => true)
      .catch(() => false);

    if (fileExists) {
      // Delete the image file
      await fs.promises.unlink(imagePath);
      res.json({ message: "Image deleted successfully" });
    } else {
      res.status(404).json({ error: "Image not found" });
    }
  } catch (error) {
    console.error("Error deleting image:", error);
    res
      .status(500)
      .json({ error: "Internal Server Error", details: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});

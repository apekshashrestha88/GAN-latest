window.onload = function () {
  fetchImage();
  const submitButton = document.getElementById("submitButton");
  submitButton.addEventListener("click", async function (event) {
    event.preventDefault();
    await uploadImage(event);
  });
  submitButton.disabled = false;

  const fileInput = document.getElementById("imageInput");

  fileInput.addEventListener("change", async function () {
    if (fileInput.files.length > 0) {
      await uploadImage();
    }
  });

  document
    .getElementById("uploadButton")
    .addEventListener("click", function () {
      fileInput.click();
    });

  async function uploadImage() {
    document.getElementById("uploadButton").hidden = true;
    document.getElementById("submitButton").hidden = true;
    document.getElementById("spinner").hidden = false;
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:3000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Image uploaded successfully:", data);

        // Enable the submit button after successful upload
        document.getElementById("submitButton").disabled = false;
      } else {
        console.error("Error uploading image:", response.status);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
    }

    // Prevent default form submission
    return false;
  }

  async function fetchImage() {
    const finalImg = document.getElementById("finalImg");

    try {
      const response = await fetch("http://localhost:3000/image");

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        finalImg.src = imageUrl;
        finalImg.hidden = false;
        document.getElementById("initialImg").hidden = false;
        document.getElementById("resetBtn").hidden = false;
        document.getElementById("submitButton").hidden = true;
        document.getElementById("uploadButton").hidden = true;
      } else {
        throw new Error("Error fetching image");
      }
    } catch (error) {
      console.error("Error fetching image:", error);
      // Retry fetching after a delay
      setTimeout(fetchImage, 2000); // Retry after 2 seconds
    }
  }

  document.getElementById("resetBtn").addEventListener("click", async () => {
    try {
      const response = await fetch("http://localhost:3000/delete-image", {
        method: "DELETE",
      });

      console.log(response.status); // Log the response status

      if (response.ok) {
        console.log("Image deleted successfully");
      } else {
        throw new Error("Error deleting image");
      }
    } catch (error) {
      console.error("Error deleting image:", error);
    }
  });
};

const cv = () => {
  const fileInput = document.getElementById("fileInput");
  const results = document.getElementById("results");
  if (!fileInput.files.length) {
          alert("Please select a file first");
          return;
      }
  
      const file = fileInput.files[0];
      console.log(file);
  
  let formData = new FormData();
  formData.append("image", fileInput.files[0]);

  fetch("/predict", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      console.log(data)
      const key = Object.keys(data[0])[0];
      const value = data[0][key];
      const img = document.createElement("img");
      img.src = "data:image/jpeg;base64," + value;
      results.appendChild(img);
    })
    .catch(err => {
      console.log(err);
    });
  
}
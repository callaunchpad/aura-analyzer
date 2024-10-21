import './App.css'
import { useState } from "react";

function App() {
  const [imgSrc, setImgSrc] = useState('');
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      const file = files[0]; // Get the uploaded file
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target && typeof event.target.result === 'string') {
          setImgSrc(event.target.result); // Set the image source to display
        }
      };
      reader.readAsDataURL(file); // Read the file as a data URL
    }
  };


  return (
    <>
      <h1>Upload an image to detect a face!</h1>
      <div className="card">
        <form id="uploadForm">
          <input type ="file" id="fileInput" accept="image/*" onChange={handleChange}/>
          <button type="submit">Upload</button>
        </form>
        {imgSrc && <img id="uploadedImage" alt="Uploaded Preview" src={imgSrc} style={{maxWidth: '1000px'}} />}
      </div>
    </>
  );
}

export default App

import './App.css'
import { useState, useEffect } from "react";
//import axios from 'axios'

function App() {
  const [imgSrc, setImgSrc] = useState('');
  const [newImgSrc, setNewImgSrc] = useState('');
  const [imageUploaded, setImageUploaded] = useState(false);

  useEffect(() => {
    // Fetch the image from the backend (or static folder)
    if (imageUploaded) {
    fetch('/cropped.jpg') // or fetch('http://localhost:3001/image') if using a server
      .then((response) => {
        if (response.ok) {
          setNewImgSrc('/cropped.jpg'); // Set the URL path to the image
        }
      })
      .catch((err) => {
        console.error('Failed to fetch image', err);
      });
      setImageUploaded(false);
    }
  }, [imageUploaded]);
  
  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      const file = files[0]; // Get the uploaded file
      const reader = new FileReader();
          // Create a FormData object
    // const formData = new FormData();
    // formData.append('file', file); // Append the file to FormData

    // // Send the file to the backend using Axios
    // try {
    //   const response = await axios.post('http://localhost:3001/upload', formData, {
    //     headers: {
    //       'Content-Type': 'multipart/form-data', // Set the appropriate content type for file upload
    //     },
    //   });
    //   console.log('File uploaded successfully:', response.data);
    // } catch (error) {
    //   console.error('Error uploading file:', error);
    // }
      reader.onload = (event) => {
        if (event.target && typeof event.target.result === 'string') {
          setImgSrc(event.target.result); // Set the image source to display
        }
      };
      reader.readAsDataURL(file); // Read the file as a data URL

      setImageUploaded(true); // Set to true to trigger useEffect
      // save file into folder
      // now put this file into ../../combined_demo/input-imgs
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
        <div className="card">{newImgSrc ? (
        <img src={newImgSrc} alt="Facial Detection Result" style={{maxWidth: '1000px'}} />
      ) : (
        <p> Awaiting upload... </p>
      )}      </div>
         </div>
    </>
  );
}

export default App

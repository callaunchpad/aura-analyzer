import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState, useEffect } from "react";
import NavBar from './components/NavBar';
import About from './pages/About';

function App() {
  const [imgSrc, setImgSrc] = useState('');
  const [newImgSrc, setNewImgSrc] = useState('');
  const [imageUploaded, setImageUploaded] = useState(false);

  useEffect(() => {
    // Fetch the image from the backend (or static folder)
    if (imageUploaded) {
    fetch('../combined-demo/output-imgs/cropped.jpg') // or fetch('http://localhost:3001/image') if using a server
      .then((response) => {
        if (response.ok) {
          setNewImgSrc('../combined-demo/output-imgs/cropped.jpg'); // Set the URL path to the image
        }
      })
      .catch((err) => {
        console.error('Failed to fetch image', err);
      });
      setImageUploaded(false);
    }
  }, [imageUploaded]);
  
  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    // navigate('/about')
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

      setImageUploaded(true); // Set to true to trigger useEffect
      // save file into folder
      // now put this file into ../../combined-demo/input-imgs
    }
  };

  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={
          <main style={{ paddingTop: '80px', backgroundColor: 'black' }}>
            <h1 className = "textsss">aura analyzer</h1>
            <div className="card">
              <form id="uploadForm" className="up">
                <label htmlFor="fileInput" className="file-input-label">
                  choose image to auralyze
                </label>
                <input type="file" id="fileInput" accept="image/*" onChange={handleChange}/>
                <button type="submit" className="auralyze">auralyze</button>
              </form>
              {imgSrc && <img id="uploadedImage" alt="Uploaded Preview" src={imgSrc} style={{maxWidth: '1000px', padding: '3.2em'}} />}
              {/* <div className="card">
                {newImgSrc ? (
                  <img src={newImgSrc} alt="Facial Detection Result" style={{maxWidth: '1000px'}} />
                ) : (
                  <p>waiting...</p>
                )}
              </div> */}
            </div>
          </main>
        } />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  );
}

export default App
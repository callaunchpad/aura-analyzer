import { useState, useEffect } from 'react';
import styles from './Home.module.css';

const Home = () => {
  const [imgSrc, setImgSrc] = useState('');
  const [newImgSrc, setNewImgSrc] = useState('');
  const [imageUploaded, setImageUploaded] = useState(false);

  useEffect(() => {
    if (imageUploaded) {
      fetch('../combined-demo/output-imgs/cropped.jpg')
        .then((response) => {
          if (response.ok) {
            setNewImgSrc('../combined-demo/output-imgs/cropped.jpg');
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
      const file = files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target && typeof event.target.result === 'string') {
          setImgSrc(event.target.result);
        }
      };
      reader.readAsDataURL(file);
      setImageUploaded(true);
    }
  };

  return (
    <div className={styles.homeContainer}>
      <h1 className={styles.title}>upload an image to detect a face!</h1>
      <div className={styles.card}>
        <form id="uploadForm">
          <input type="file" id="fileInput" accept="image/*" onChange={handleChange}/>
          <button type="submit">upload</button>
        </form>
        {imgSrc && <img id="uploadedImage" alt="uploaded preview" src={imgSrc} className={styles.image} />}
        <div className={styles.card}>
          {newImgSrc ? (
            <img src={newImgSrc} alt="facial detection result" className={styles.image} />
          ) : (
            <p>awaiting upload...</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;

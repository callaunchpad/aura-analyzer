import styles from './About.module.css';

const About = () => {
  return (
    <div className={styles.aboutContainer}>
      <h1 className={styles.title}>aura analyzer</h1>
      <div className={styles.content}>
        
        <p>
          aura analyzer is an innovative tool that helps you understand the emotional
          and energetic qualities captured in your photos. using advanced facial
          detection and analysis, we reveal the subtle energies and auras present
          in your images.
        </p>
        <p>
          our technology combines modern computer vision with traditional aura
          interpretation techniques to provide you with unique insights about the
          energy fields present in your photographs.
        </p>
      </div>
      <div className={styles.facesGrid}>
        <div className={styles.facesRow}>
          <div>
            <img src="jade.png" className={styles.roundImage} alt="Jade" />
            <p className={styles.faceName}>jade</p>
          </div>
          <div>
            <img src="kathryn.png" className={styles.roundImage} alt="Kathryn" />
            <p className={styles.faceName}>kathryn</p>
          </div>
          <div>
            <img src="anaisha.png" className={styles.roundImage} alt="Anaisha" />
            <p className={styles.faceName}>anaisha</p>
          </div>
          <div>
            <img src="andrea.png" className={styles.roundImage} alt="Andrea" />
            <p className={styles.faceName}>andrea</p>
          </div>
          <div>
            <img src="seb.png" className={styles.roundImage} alt="Seb" />
            <p className={styles.faceName}>sebastian</p>
          </div>
        </div>
        <div className={styles.facesRow}>
          <div>
            <img src="spencer.png" className={styles.roundImage} alt="Spencer" />
            <p className={styles.faceName}>spencer</p>
          </div>
          <div>
            <img src="sophie.png" className={styles.roundImage} alt="Sophie" />
            <p className={styles.faceName}>sophie</p>
          </div>
          <div>
            <img src="mischa.png" className={styles.roundImage} alt="Mischa" />
            <p className={styles.faceName}>mischa</p>
          </div>
          <div>
            <img src="iris.png" className={styles.roundImage} alt="Iris" />
            <p className={styles.faceName}>iris</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;

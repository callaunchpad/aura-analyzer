import styles from './NavBar.module.css';
import { Link } from 'react-router-dom';
import logo from '../assets/aura_analyzer_logo.png';

const NavBar = () => {
  return (
    <nav className={styles.navbar}>
      <div className={styles.logo}>
        <Link to="/" className={styles.logoLink}>
          <img src={logo} alt="Aura Analyzer Logo" className={styles.logoImage} />
        </Link>
      </div>
      <ul className={styles.navLinks}>
        <li><Link to="/" className={styles.navLink}>home</Link></li>
        <li><Link to="/about" className={styles.navLink}>about</Link></li>
        <li><Link to="/results" className={styles.navLink}>results</Link></li>
      </ul>
    </nav>
  );
};

export default NavBar;

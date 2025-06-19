import Logo from "../assets/Logo.svg";
import "../styles/components/Navbar.scss";

const Navbar = () => {
    return (
        <div className="Navbar">
            <div className="Logo">
                <img src={Logo} alt="Logo" />
            </div>
        </div>
    )
}

export default Navbar